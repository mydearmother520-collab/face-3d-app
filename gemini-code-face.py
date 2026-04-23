import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os

# 標準導入
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

st.title("AI 3D 人臉掃描器")

# --- 只要這兩行，雲端就會自動處理 ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 1. 取得這支程式所在的資料夾
base_path = os.path.dirname(os.path.abspath(__file__))

# 2. 定義虛擬環境中套件的「絕對路徑」
# 這樣不管 Python 怎麼迷路，我們都直接把它帶到套件面前
venv_site_packages = os.path.join(base_path, "venv", "Lib", "site-packages")

if os.path.exists(venv_site_packages):
    # 強制把虛擬環境的路徑放到最優先順序
    sys.path.insert(0, venv_site_packages)
else:
    st.error(f"找不到虛擬環境資料夾：{venv_site_packages}")
    st.info("請確保 Face3D_Share 資料夾內有 venv 資料夾。")
    st.stop()

# 3. 現在嘗試載入，如果還是失敗，代表 venv 裡的 mediapipe 沒裝好
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh
    mp_face_mesh = face_mesh
    st.sidebar.success("✅ AI 引擎連接成功")
except Exception as e:
    st.error(f"AI 引擎啟動失敗。錯誤訊息：{e}")
    st.info("建議：請在 CMD 執行 'venv\\Scripts\\pip install mediapipe==0.10.14' 重裝。")
    st.stop()

import numpy as np
import PIL.Image as Image
import plotly.graph_objects as go

import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image
import plotly.graph_objects as go

# 1. 初始化 MediaPipe Face Mesh (AI 核心)
mp_face_mesh = mp.solutions.face_mesh

# 設定網頁介面
st.set_page_config(page_title="AI 3D Face Scanner", layout="wide")
st.title("🛡️ 3D 人臉虛擬模型重建系統")
st.markdown("---")

# 側邊欄控制面板
st.sidebar.header("⚙️ 運算參數調整")
point_size = st.sidebar.slider("點雲顆粒大小", 1, 10, 3)
z_scale = st.sidebar.slider("深度感強化 (Z-Axis)", 0.1, 5.0, 1.5)

# 2. 檔案上傳
uploaded_file = st.file_uploader("請上傳人臉照片 (建議正臉、光線充足)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 讀取圖片
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape

    # 3. 啟動 AI 運算
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(img_array)

    if results.multi_face_landmarks:
        st.success("✅ 成功辨識臉部！正在生成 3D 彩色點雲...")
        
        pts = []
        colors = []
        
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                # 數學座標轉換：X, Y 映射至像素，Z 軸根據深度強化
                # 為了視覺直觀，Z 軸取負值處理
                pts.append([lm.x * w, -lm.y * h, -lm.z * w * z_scale])
                
                # 獲取像素顏色
                ix, iy = int(lm.x * w), int(lm.y * h)
                ix, iy = np.clip(ix, 0, w-1), np.clip(iy, 0, h-1)
                r, g, b = img_array[iy, ix]
                colors.append(f'rgb({r},{g},{b})')

        pts_np = np.array(pts)

        # 4. 使用 Plotly 進行 3D 渲染 (最穩定、不挑顯卡)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="原始圖片", use_container_width=True)
            
        with col2:
            st.write("### 3D 運算結果")
            fig = go.Figure(data=[go.Scatter3d(
                x=pts_np[:, 0],
                y=pts_np[:, 1],
                z=pts_np[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=colors,
                    opacity=1.0
                )
            )])

            # 設定 3D 視角與黑色背景
            fig.update_layout(
                scene=dict(
                    bgcolor='black',
                    xaxis_visible=False,
                    yaxis_visible=False,
                    zaxis_visible=False,
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='black'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.info("💡 操作指南：按住左鍵旋轉模型，滾輪縮放。")
    else:
        st.error("⚠️ 無法偵測到臉部，請確保照片清晰且無遮擋。")
