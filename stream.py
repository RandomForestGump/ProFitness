import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
from Utils.Visualizer import show_image
from mediapipe.python.solutions import pose as mp_pose
from ParseClassifier import avengers_assemble

font = cv2.FONT_HERSHEY_SIMPLEX
test = False
# org
timeout = 60
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

pose_tracker, pose_embedder, pose_classifier, pose_classification_filter, repetition_counter, pose_classification_visualizer = avengers_assemble(0)
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'
z = 0
st.title('Pose Corrector AI Trainer Application using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('ProFitness')
st.sidebar.subheader('Parameters')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['Shoulder Press', 'About App']
                                )

if app_mode == 'About App':

    st.markdown(
        'In this application we are using **MediaPipe** for Counting Repetitions and Correcting pose. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.video('https://www.youtube.com/watch?v=FMaNNXgB_5c&ab_channel=AugmentedStartups')

    st.markdown('''
          # About Me \n 
            My name is ** Rajat Jain ** . \n
            ''')






elif app_mode == 'Shoulder Press':

    start = time.time()

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    storage_queue = []
    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    use_webcam= True
    if not video_file_buffer:
        if use_webcam:
            print('Using_Webcam')
            vid = cv2.VideoCapture(0)
        else:
            pass
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))
    print('Done till here')
    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Repititions**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")
    with kpi4:
        st.markdown("**Wrong Counter**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_pose.Pose(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as face_mesh:
        prevTime = 0
        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result = pose_tracker.process(image=frame)
            pose_landmarks = result.pose_landmarks

            # Draw pose prediction.
            output_frame = frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            if pose_landmarks is not None:

                # Get landmarks.
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)

                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks)
                pose_classification_filtered = pose_classification_filter(pose_classification)
                print(pose_classification_filtered)
                # Count repetitions.
                test, repetitions_count = repetition_counter(pose_classification_filtered)
                print(repetitions_count)
                outx = pose_classification_visualizer(
                    frame=output_frame,
                    pose_classification=pose_classification,
                    pose_classification_filtered=pose_classification_filtered,
                    repetitions_count=repetitions_count)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                out.write(frame)
            # Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{repetitions_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(output_frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=output_frame, width=640)
            storage_queue.append(output_frame)
            stframe.image(output_frame, channels='BGR', use_column_width=True)

            end = time.time()

            if end - start > timeout:
                vid.release()
                cv2.destroyAllWindows()
                break

    st.text('Video Processed')

    # output_video = open('output1.mp4', 'rb')
    # out_bytes = output_video.read()
    # st.video(out_bytes)


    def merge(intervals):

        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # otherwise, there is overlap, so we merge the current and previous
                # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

    x = []
    class_name = 'press_down'
    for classification in pose_classification_visualizer._pose_classification_filtered_history:
        if classification is None:
            x.append(None)
        elif class_name in classification:
            x.append(classification[class_name])
        else:
            x.append(0)

    res = []

    for i in range(len(x)):
        window = 20
        thresh = 8.5
        area = x[i:i + window]
        s = [True if el > thresh else False for el in area]
        if sum(s) == len(s):
            res.append([i, i + window])
        else:
            continue

    intervals = merge(res)
    # intervals.pop()


    print(intervals)
    print(len(storage_queue))
    c = 0
    pathx = '/Users/rajatjain/Desktop/blooper_reel/'
    if len(intervals) == 0:
        print('No Errors found')

    for interval in intervals:
        fail = storage_queue[interval[0]: interval[1] + 5]
        c += 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fail_video = cv2.VideoWriter(pathx+'fail_{}.mov'.format(c), cv2.VideoWriter_fourcc(*'mp4v'), 60,
                                     (width, height))
        for j in range(len(fail)):
            fail_video.write(np.asarray(fail[j]))
        fail_video.release()

    print('Finally Done')
    vid.release()
    out.release()
    st.stop()

