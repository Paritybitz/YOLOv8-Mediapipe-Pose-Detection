from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cwd = os.getcwd()

if not os.path.exists('model'):
    os.mkdir('model')

if not os.path.exists(f'{cwd}/result'):
    os.mkdir(f'{cwd}/result')

TRAIN_DATASET_PATH = f'{cwd}/shootingplayer_new/train'
VALID_DATASET_PATH = f'{cwd}/shootingplayer_new/val'
TEST_DATASET_PATH = f'{cwd}/shootingplayer_new/test'
MODEL_PATH = f'{cwd}/model'

MODEL = 'efficientdet_lite0'
MODEL_NAME = 'shooting_player_det1.tflite'
CLASSES = ['shooting player']
EPOCHS = 20
BATCH_SIZE = 4


train_data = object_detector.DataLoader.from_pascal_voc(
    TRAIN_DATASET_PATH,
    TRAIN_DATASET_PATH,
    CLASSES)

val_data = object_detector.DataLoader.from_pascal_voc(
    VALID_DATASET_PATH,
    VALID_DATASET_PATH,
    CLASSES)

spec = model_spec.get(MODEL)

model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=BATCH_SIZE,
    train_whole_model=True,
    epochs=EPOCHS,
    validation_data=val_data
)

model.evaluate(val_data)

model.export(export_dir=MODEL_PATH, tflite_filename=MODEL_NAME)