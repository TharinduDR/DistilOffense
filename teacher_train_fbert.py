import pandas as pd
import sklearn
import torch
import os

from datasets import Dataset
from datasets import load_dataset
from deepoffense.classification import ClassificationModel
from examples.common.print_stat import print_information
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from evaluation import macro_f1, weighted_f1
from label_converter import encode, decode
from teacher_config_bert import MODEL_TYPE, MODEL_NAME, args, TEMP_DIRECTORY, RESULT_FILE

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

olid_train = pd.read_csv('data/olid_train.csv', sep="\t")
olid_test = pd.read_csv('data/olid_test.csv', sep="\t")
solid = Dataset.to_pandas(load_dataset('tharindu/SOLID', split='train', sep="\t"))

olid_test_sentences = olid_test["Text"].to_list()
solid_sentences = solid["text"].to_list()


train = pd.concat([olid_train], ignore_index=True)
train = train.rename(columns={'Text': 'text', 'Class': 'labels'})
train = train[['text', 'labels']]
train = train.sample(frac=1).reset_index(drop=True)
train['labels'] = encode(train["labels"])

model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                    use_cuda=torch.cuda.is_available(),
                                    cuda_device=1)

train_df, eval_df = train_test_split(train, test_size=0.2, random_state=args["manual_seed"])
model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                          accuracy=sklearn.metrics.accuracy_score)
model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
                                    use_cuda=torch.cuda.is_available(), cuda_device=1)

predictions, raw_outputs = model.predict(olid_test_sentences)

olid_test['predictions'] = predictions
olid_test['predictions'] = decode(olid_test['predictions'])

print_information(olid_test, "predictions", "Class")

solid_predictions, solid_raw_outputs = model.predict(solid_sentences)

probability_predictions = []

for output in solid_raw_outputs:
    weights = softmax(output)
    probability_predictions.append(weights)

solid["fbert_predictions"] = probability_predictions
prediction_file = solid[["id", "fbert_predictions"]].copy()
prediction_file.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE),  header=True, sep='\t', index=False, encoding='utf-8')

