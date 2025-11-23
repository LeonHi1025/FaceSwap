import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from gfpgan.utils import GFPGANer

# ✅ 主角選擇模組（embedding only）
class TargetSelector:
    def __init__(self, app, target_img, threshold=0.4):
        self.app = app
        self.threshold = threshold
        img = cv2.imread(target_img)
        if img is None:
            raise FileNotFoundError(f"❌ 無法讀取 target 圖片：{target_img}")
        target_faces = self.app.get(img)
        if not target_faces:
            raise ValueError("❌ 無法在 target 圖片中偵測到人臉")
        self.target_embedding = target_faces[0].embedding

    def select(self, faces):
        best_score, best_face = -1, None
        for face in faces:
            score = self._cosine(face.embedding, self.target_embedding)
            if score > best_score:
                best_score, best_face = score, face
        return best_face if best_score > self.threshold else None

    def _cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ✅ 換臉流程
def swap_main_character(input_path, source_path, target_path, output_path, swapper_model, threshold):
    print("🚀 啟動 InsightFace 模型...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("🔁 載入 InSwapper 模型:", swapper_model)
    swapper = model_zoo.get_model(swapper_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    print("📸 讀取 source 臉部圖片...")
    source_img = cv2.imread(source_path)
    if source_img is None:
        raise FileNotFoundError(f"❌ 無法讀取 source 圖片：{source_path}")
    source_img = cv2.convertScaleAbs(source_img, alpha=1.2, beta=30)
    source_faces = app.get(source_img)
    if not source_faces:
        raise ValueError("❌ 無法在 source 圖片中偵測到人臉")
    source_face = source_faces[0]

    selector = TargetSelector(app, target_path, threshold=threshold)

    print("🎥 開始換臉處理:", input_path)
    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="換臉中", unit="幀"):
        ret, frame = video.read()
        if not ret or frame is None:
            continue
        faces = app.get(frame)
        main_face = selector.select(faces) if faces else None
        if main_face:
            swapped = swapper.get(frame, main_face, source_face)
            if swapped is not None and swapped.shape == frame.shape:
                frame = swapped
        out.write(frame)

    video.release()
    out.release()
    print("✅ 主角換臉完成，輸出為", output_path)

# ✅ GFPGAN 修復流程
def enhance_video(input_path, output_path):
    print("🧠 啟動 GFPGAN 修復流程...")
    model_path = 'gfpgan/weights/GFPGANv1.4.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GFPGAN 模型不存在: {model_path}")
    enhancer = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="GFPGAN 修復中", unit="幀"):
        ret, frame = video.read()
        if not ret or frame is None:
            continue
        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_crop = frame[y1:y2, x1:x2]
            try:
                _, _, restored = enhancer.enhance(face_crop, has_aligned=False, only_center_face=True, paste_back=True)
                if restored is not None and restored.shape == face_crop.shape:
                    frame[y1:y2, x1:x2] = restored
            except Exception:
                continue
        out.write(frame)

    video.release()
    out.release()
    print("✅ GFPGAN 修復完成，輸出為", output_path)

# ✅ 使用 FFmpeg 合併音訊
def merge_audio(video_path, audio_source_path, output_path):
    print(f"🎵 使用 FFmpeg 合併音訊到 {output_path}")
    cmd = f'ffmpeg -y -i "{video_path}" -i "{audio_source_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "{output_path}"'
    os.system(cmd)

# ✅ 主流程
def main(args):
    base_name = os.path.splitext(args.output)[0]
    raw_video = f"{base_name}_raw.mp4"
    enhanced_video = f"{base_name}_enhanced.mp4"

    swap_main_character(
        input_path=args.input,
        source_path=args.source,
        target_path=args.target,
        output_path='temp_raw.mp4',
        swapper_model=args.swapper,
        threshold=args.threshold
    )

    merge_audio('temp_raw.mp4', args.input, raw_video)

    enhance_video('temp_raw.mp4', 'temp_enhanced.mp4')
    merge_audio('temp_enhanced.mp4', args.input, enhanced_video)

    os.remove('temp_raw.mp4')
    os.remove('temp_enhanced.mp4')

    print(f"\n✅ 已完成兩部影片輸出：\n- 未修復版本：{raw_video}\n- 修復版本：{enhanced_video}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='要換上去的臉部圖片')
    parser.add_argument('--target', required=True, help='主角臉部圖片（用來選主角）')
    parser.add_argument('--input', required=True, help='原始影片')
    parser.add_argument('--output', required=True, help='輸出影片名稱（不含 _raw/_enhanced）')
    parser.add_argument('--threshold', type=float, default=0.3, help='主角辨識的相似度門檻（預設 0.2）')
    parser.add_argument('--swapper', default='inswapper_128.onnx', help='換臉模型路徑')
    args = parser.parse_args()
    main(args)
