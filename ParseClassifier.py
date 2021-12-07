# Initilize tracker, classifier and counter.
# Do that before every video as all of them have state.
from mediapipe.python.solutions import pose as mp_pose
from Classifier.PoseClassifier import PoseClassifier
from Classifier.PoseEmbedder import FullBodyPoseEmbedder
from Utils.Smooth import EMADictSmoothing
from Utils.RepCounter import RepetitionCounter
from Utils.Visualizer import PoseClassificationVisualizer

# Folder with pose class CSVs. That should be the same folder you using while
# building classifier to output CSVs.


def avengers_assemble(video_n_frames):
    path = '/Users/rajatjain/Desktop/workout_images/'
    pose_samples_folder = path+'fitness_poses_csvs_out'
    video_path = '/Users/rajatjain/Desktop/vid1.mov'
    class_name='press_down'
    out_video_path = '/Users/rajatjain/Desktop/press-sample-out1.mov'

    # Initialize tracker.
    pose_tracker = mp_pose.Pose()

    # Initialize embedder.
    pose_embedder = FullBodyPoseEmbedder()

    # Initialize classifier.
    # Ceck that you are using the same parameters as during bootstrapping.
    pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

    # # Uncomment to validate target poses used by classifier and find outliers.
    # outliers = pose_classifier.find_pose_sample_outliers()

    # print('Number of pose sample outliers (consider removing them): ', outliers)

    # Initialize EMA smoothing.
    pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

    # Initialize counter.
    repetition_counter = RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4)

    # Initialize renderer.
    pose_classification_visualizer = PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=video_n_frames,
    # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
    plot_y_max=10)

    return pose_tracker, pose_embedder, pose_classifier, pose_classification_filter, repetition_counter, pose_classification_visualizer