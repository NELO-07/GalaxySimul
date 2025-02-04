import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
import sys
import time
from mpi4py import MPI
import tqdm

# 커맨드라인 인자로부터 디렉터리 경로 얻기
try:
    dir_path = sys.argv[1]
except IndexError:
    dir_path = "outputs"

resolution = 1

# 디렉터리 내 모든 .npy 파일을 찾아 정렬
frame_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.npy')])

def make_frame_image(file_path, index, resolution):
    data = np.load(file_path, allow_pickle=True)

    fig = plt.figure(figsize=(16 * resolution, 9 * resolution))
    ax = plt.axes(projection='3d')

    # 배경색 설정
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # 축 눈금 제거
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 축 눈금 및 레이블 색상 지정
    ax.tick_params(colors='white', labelcolor='white')

    # 회색 축 면 및 윤곽선 제거
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # 축 범위 및 시점 설정
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)
    ax.view_init(elev=0, azim=0)

    # 은하별 별 색상을 20% 노란색, 80% 흰색으로 랜덤 지정
    nstars1 = data[0].shape[1]
    nstars2 = data[1].shape[1]

    # 은하1 색상 설정
    randvals1 = np.random.rand(nstars1) < 0.2  # 20% 확률로 True
    colors1 = np.ones((nstars1, 3))            # 기본 흰색
    colors1[randvals1] = [1, 1, 0]             # 20%를 노란색으로

    # 은하2 색상 설정
    randvals2 = np.random.rand(nstars2) < 0.2
    colors2 = np.ones((nstars2, 3))
    colors2[randvals2] = [1, 1, 0]

    # 은하1과 은하2의 별 랜덤 색상으로 표시
    ax.scatter3D(data[0][0, :], data[0][1, :], data[0][2, :], s=0.3, c=colors1)
    ax.scatter3D(data[1][0, :], data[1][1, :], data[1][2, :], s=0.3, c=colors2)

    # [은하 중심 계산 / annotate 부분 제거 → 마커 없음]

    # 제목 설정
    ax.set_title(f'Frame {index}', color='white')

    # 그림을 그려서 RGBA 배열로 추출
    fig.canvas.draw()
    image_arr = np.array(fig.canvas.buffer_rgba())
    pil_img = Image.fromarray(image_arr).resize((1920, 1080))
    plt.close(fig)
    return pil_img


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.time()

    num_frames = len(frame_files)

    # 균등 분할(나머지는 앞의 rank부터 1개씩 추가)
    frames_per_rank = num_frames // size
    remainder = num_frames % size

    local_start = rank * frames_per_rank + min(rank, remainder)
    local_count = frames_per_rank + (1 if rank < remainder else 0)
    local_end = local_start + local_count

    local_files = frame_files[local_start:local_end]
    # 실제 프레임 번호(1부터 시작)
    local_indices = range(local_start + 1, local_end + 1)

    # rank별 임시 PNG 저장 디렉터리
    temp_dir = f"temp_frames_{rank}"
    os.makedirs(temp_dir, exist_ok=True)

    # 진행률 표시
    pbar = tqdm.tqdm(
        zip(local_files, local_indices),
        desc=f"Rank {rank} rendering frames",
        total=len(local_files),
        position=rank
    )

    local_results = []
    for file_path, frame_idx in pbar:
        # 렌더링 (마커 없음)
        pil_img = make_frame_image(file_path, frame_idx, resolution)
        # 임시 PNG 파일 경로
        png_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
        pil_img.save(png_path)

        # (index, png_path)만 보관
        local_results.append((frame_idx, png_path))

    # 모든 rank에서 local_results를 모아, rank=0이 정렬/최종 동영상 생성
    all_data = comm.gather(local_results, root=0)

    if rank == 0:
        # 루트에서 병합
        merged = []
        for chunk in all_data:
            merged.extend(chunk)
        # 인덱스 순으로 정렬
        merged.sort(key=lambda x: x[0])

        # MP4 생성
        pbar_movie = tqdm.tqdm(merged, desc="Making Movie", position=size)
        output_file = f"{dir_path}_mov.mp4"
        writer = imageio.get_writer(output_file, fps=60, macro_block_size=None, format='MP4')

        for frame_idx, png_path in pbar_movie:
            img_arr = imageio.v2.imread(png_path)
            writer.append_data(img_arr)
        writer.close()

        print(f"The movie saved as {output_file}")

        end_time = time.time()
        print(f"\(^_^)/ \(^_^)/ Total time for making the movie: {end_time - start_time:.3f} seconds.")

        # (선택) 임시 디렉터리/파일 삭제
        # import shutil
        # for r in range(size):
        #     shutil.rmtree(f"temp_frames_{r}")
