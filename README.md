# 은하 충돌 시뮬레이션

## 개요
이 프로젝트는 두 개의 은하가 충돌하는 과정을 병렬 컴퓨팅(MPI)을 이용하여 시뮬레이션합니다. 시뮬레이션은 별들의 움직임과 은하 간 상호작용을 추적하며, 각 타임스텝마다 스냅샷을 저장합니다.

## 주요 기능
- MPI(Message Passing Interface)를 사용한 병렬 컴퓨팅
- 초기 조건(은하의 방향, 질량 비율 등) 사용자 지정 가능
- 동적 마찰 및 중력 상호작용 포함
- 각 스텝마다 별들의 위치 데이터를 저장

## 요구 사항
- Python 3.x
- NumPy
- mpi4py

## 설치
```sh
pip install numpy mpi4py
```

## 사용법
기본 설정으로 시뮬레이션 실행:
```sh
mpirun -np <프로세스 개수> python main.py
```

사용자 정의 설정으로 실행:
```sh
mpirun -np <프로세스 개수> python main.py --theta1 30 --phi1 300 --theta2 45 --phi2 60 --tot_nstar 20000 --mratio 2.0 --peri 6.0 --dt 0.05 --nstep 1000 --big_halo --seed_fix
```

### 명령어 옵션
- `--theta1`, `--phi1`: 은하 1의 방향 설정
- `--theta2`, `--phi2`: 은하 2의 방향 설정
- `--tot_nstar`: 총 별 개수
- `--mratio`: 두 은하의 질량 비율
- `--peri`: 최근접 거리(근일점 거리)
- `--dt`: 시간 간격 크기
- `--nstep`: 시뮬레이션 스텝 개수
- `--big_halo`: 더 큰 암흑 물질 헤일로 사용
- `--seed_fix`: 동일한 난수 시드를 사용하여 재현 가능하게 설정

## 출력 데이터
시뮬레이션 결과는 `outputs/` 디렉터리에 `.npy` 파일로 저장되며, 각 파일은 별들의 위치 스냅샷을 포함합니다.

## 제작자
Nelo

