import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import numpy as np

# ==========================================
# 0. 경로 설정 및 하이퍼파라미터
# ==========================================
# Kaggle에서 다운로드한 'handwritten-digits-dataset-not-in-mnist'의 추출된 폴더 경로를 입력하세요.
# 구조 예시: ./not_in_mnist/0, ./not_in_mnist/1, ..., ./not_in_mnist/6, ...
NOT_IN_MNIST_DIR = './handwritten-digits-dataset-not-in-mnist' 
BATCH_SIZE = 64
EPOCHS = 10  # 데이터가 늘어났으므로 조금 더 충분히 학습합니다.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 데이터 증강 및 데이터셋 통합 파이프라인
# ==========================================
# 6과 8의 형태가 왜곡에 의해 서로 겹치지 않도록 ElasticTransform 강도를 소폭 조절
# Gaussian Noise를 추가하기 위한 커스텀 트랜스폼 클래스 정의
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        # 0~1 사이의 텐서 데이터에 미세한 노이즈 추가
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0) # 0~1 범위를 벗어나지 않도록 클리핑

# 정밀하게 튜닝된 6/8 분류 전용 증강 파이프라인
train_transform = transforms.Compose([
    # 1. Rotation & Affine: 6과 8의 기울어짐과 위치 다양성 확보
    transforms.RandomRotation(degrees=12), # 과도한 회전은 6과 9를 헷갈리게 하므로 12도로 소폭 제한
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)), 
    
    # 2. Perspective (원근 변형): 필기할 때 종이가 놓인 각도 모사
    # distortion_scale이 너무 크면 글씨가 찌그러지므로 0.2 내외가 안전합니다.
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5), 
    
    # 3. ElasticTransform (탄성 왜곡): 손가락 힘 조절에 따른 선의 굵기/휨 모사
    transforms.ElasticTransform(alpha=30.0, sigma=5.0), 
    
    # 4. Blur (흐림 효과): 카메라 초점이 흐려지거나 잉크가 번진 효과
    # 커널 크기는 3x3 고정, 시그마 값을 아주 작게 주어 글씨 형태가 파괴되지 않게 합니다.
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.8))
    ], p=0.5),
    
    # 텐서 변환 (이 시점에서 데이터 범위가 0~1이 됨)
    transforms.ToTensor(),
    
    # 5. Gaussian Noise: 미세한 지직거림(센서 노이즈) 추가
    AddGaussianNoise(mean=0.0, std=0.03) # std가 0.05를 넘으면 숫자가 안 보일 수 있음
])

# 1) 기본 MNIST 데이터셋
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=train_transform)

# 2) EMNIST 데이터셋 (Digits 클래스로 0~9 손글씨 대량 보강)
emnist_train = datasets.EMNIST('./data', split='digits', train=True, download=True, transform=train_transform)

# 3) Not in MNIST 데이터셋 (Kaggle)
# ImageFolder는 기본 3채널(RGB)로 읽으므로, Grayscale 변환 후 ToTensor를 적용해야 합니다.
kaggle_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)), # 혹시 모를 이미지 크기 불일치 방지
    train_transform
])

# Kaggle 데이터셋 폴더가 존재하는 경우에만 로드 (예외 처리)
if os.path.exists(NOT_IN_MNIST_DIR):
    kaggle_train = datasets.ImageFolder(root=NOT_IN_MNIST_DIR, transform=kaggle_transform)
    # 3개 데이터셋 통합
    full_dataset = ConcatDataset([mnist_train, emnist_train, kaggle_train])
    print(f"Successfully integrated 3 datasets! Total size: {len(full_dataset)}")
else:
    print(f"[Warning] '{NOT_IN_MNIST_DIR}' 경로를 찾을 수 없습니다. MNIST + EMNIST 조합으로만 진행합니다.")
    full_dataset = ConcatDataset([mnist_train, emnist_train])

# ==========================================
# 2. 6번과 8번에 집중하기 위한 샘플러 및 가중치 설정
# ==========================================
print("Calculating sample weights for focused learning on 6 and 8...")
targets = []

# ConcatDataset 내부의 모든 정답 라벨(target)을 수집합니다.
for dataset in full_dataset.datasets:
    if isinstance(dataset, datasets.MNIST) or isinstance(dataset, datasets.EMNIST):
        targets.extend(dataset.targets.tolist())
    elif isinstance(dataset, datasets.ImageFolder):
        targets.extend(dataset.targets)

targets = np.array(targets)

# 클래스별 빈도 계산
class_counts = np.bincount(targets, minlength=10)
print(f"Data Distribution per class: {class_counts}")

# 6과 8을 더 자주 보게 만들기 위해 샘플링 가중치 부여 (기본 가중치의 2배 부여)
class_weights_for_sampling = 1.0 / class_counts
class_weights_for_sampling[6] *= 2.0  # 6번 부스팅
class_weights_for_sampling[8] *= 2.0  # 8번 부스팅

sample_weights = class_weights_for_sampling[targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader 세팅 (sampler를 쓸 때는 shuffle=True를 지워야 합니다)
train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=sampler)

# 손실 함수(Loss) 단계에서도 6과 8을 틀렸을 때 페널티를 더 강하게 부여합니다.
# 6번과 8번의 Loss 가중치를 1.5배로 세팅
loss_weights = torch.ones(10, dtype=torch.float32).to(device)
loss_weights[6] = 1.5
loss_weights[8] = 1.5
criterion = nn.CrossEntropyLoss(weight=loss_weights)


# ==========================================
# 3. 기존 C++ mnistCUDNN 일치 모델 정의 (유지)
# ==========================================
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.view(-1, 50 * 4 * 4)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================
# 4. 학습 루틴 실행
# ==========================================
model = MnistNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining started with Enhanced Dataset & Focus on '6' and '8'...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct_6_8 = 0
    total_6_8 = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 실시간으로 6과 8의 학습 정확도 모니터링을 위한 코드
        pred = output.argmax(dim=1, keepdim=True)
        mask = (target == 6) | (target == 8)
        correct_6_8 += pred[mask].eq(target[mask].view_as(pred[mask])).sum().item()
        total_6_8 += mask.sum().item()
        
    epoch_loss = running_loss / len(train_loader)
    acc_6_8 = (correct_6_8 / total_6_8 * 100) if total_6_8 > 0 else 0
    print(f"Epoch {epoch}/{EPOCHS} Completed. Loss: {epoch_loss:.4f} | Train Acc on (6, 8): {acc_6_8:.2f}%")


# ==========================================
# 5. C++ cuDNN 호환 바이너리 파일 내보내기 (유지)
# ==========================================
def export_to_binary(tensor, filename):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    data.tofile(filename)
    print(f"Saved: {filename} (Shape: {data.shape})")

print("\nExporting trained weights to binary files...")
state_dict = model.state_dict()

export_to_binary(state_dict['conv1.weight'], 'my_conv1.bin')
export_to_binary(state_dict['conv1.bias'], 'my_conv1.bias.bin')
export_to_binary(state_dict['conv2.weight'], 'my_conv2.bin')
export_to_binary(state_dict['conv2.bias'], 'my_conv2.bias.bin')
export_to_binary(state_dict['fc1.weight'].t(), 'my_ip1.bin')
export_to_binary(state_dict['fc1.bias'], 'my_ip1.bias.bin')
export_to_binary(state_dict['fc2.weight'].t(), 'my_ip2.bin')
export_to_binary(state_dict['fc2.bias'], 'my_ip2.bias.bin')

print("All weights are successfully exported to .bin files!")