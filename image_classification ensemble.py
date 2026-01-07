import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import RandomOverSampler
import gc
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA

# === SETUP ===
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

# === DIRECTORIES ===
data_dir = "/media/igi/igi_ai_storage/mnt/IGI-AI-Files/all_data_LG/LG_2025/MJJ"
os.makedirs("models_full", exist_ok=True)
os.makedirs("features_full", exist_ok=True)
os.makedirs("plots_full", exist_ok=True)

# === CLASSES ===
classes = ['D', 'E', 'F', 'G', 'H']

# === ADVANCED AUGMENTATION ===
def get_augmentation():
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# === CUSTOM DATASET CLASS ===
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, self.targets[idx]

# === LOAD FULL DATASET ===
full_dataset = datasets.ImageFolder(data_dir)
image_paths = []
targets = []
for class_idx, class_name in enumerate(full_dataset.classes):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        if os.path.isfile(image_path):
            image_paths.append(image_path)
            targets.append(class_idx)
dataset = CustomDataset(image_paths, targets, transform=get_augmentation())
print(f"Using full dataset: {len(dataset)} samples")

# === MODEL NAMES (MODERN ARCHITECTURES) ===
model_names = [
    'efficientnet_b4', 'efficientnet_b5',
    'convnext_small', 'swin_t', 'mobilenet_v3_large'
]

# === FEATURE EXTRACTOR ===
def get_feature_extractor(model_name, fine_tune=True):
    if model_name == 'efficientnet_b4':
        weights = models.EfficientNet_B4_Weights.DEFAULT
    elif model_name == 'efficientnet_b5':
        weights = models.EfficientNet_B5_Weights.DEFAULT
    elif model_name == 'convnext_small':
        weights = models.ConvNeXt_Small_Weights.DEFAULT
    elif model_name == 'swin_t':
        weights = models.Swin_T_Weights.DEFAULT
    elif model_name == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model = getattr(models, model_name)(weights=weights).to(device)
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        elif hasattr(model, 'head'):
            model.head = nn.Identity()
        elif hasattr(model, 'fc'):
            model.fc = nn.Identity()
    model.eval()
    return model

# === EXTRACT FEATURES ===
def extract_features(loader, model):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Extracting features', leave=False):
            inputs = inputs.to(device)
            features = model(inputs)
            if len(features.shape) > 2:
                features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.vstack(all_features), np.array(all_labels, dtype=np.int64)

# === MAIN TRAINING ===
def main():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    estimators = []
    all_features = []
    all_labels = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), targets)):
        print(f'\n=== Fold {fold + 1}/5 ===')

        # Class weights for imbalanced data
        train_targets = [targets[i] for i in train_idx]
        class_weights = compute_sample_weight('balanced', train_targets)
        train_sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(train_idx), replacement=True)
        train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, num_workers=4, pin_memory=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

        # Extract features for all models
        fold_features = []
        for name in model_names:
            print(f"\nProcessing {name}...")
            model = get_feature_extractor(name)
            X_train, y_train = extract_features(train_loader, model)
            X_test, y_test = extract_features(test_loader, model)

            # Oversample minority classes
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X_train, y_train)

            # Split into train and validation sets for Optuna
            X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
            )

            # Define the objective function
            def objective(trial):
                params = {
                    'objective': 'multi:softprob',
                    'num_class': len(classes),  # Explicitly set to 5
                    'eval_metric': 'mlogloss',
                    'device': 'cpu',
                    'n_jobs': -1,
                    'random_state': 42,
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                clf = xgb.XGBClassifier(**params)
                clf.fit(X_train_opt, y_train_opt)
                preds = clf.predict_proba(X_val_opt)
                accuracy = (preds.argmax(axis=1) == y_val_opt).mean()
                return accuracy

            # Optuna hyperparameter tuning
            study = optuna.create_study(direction='maximize', sampler=TPESampler())
            study.optimize(objective, n_trials=10)
            best_params = study.best_params
            clf = xgb.XGBClassifier(**best_params)
            clf.fit(X_res, y_res)
            joblib.dump(clf, f'models_full/{name}_fold{fold}.pkl')

            # Append features for this model
            fold_features.append(X_test)
            estimators.append((f"{name}_fold{fold}", clf))

        # Stack features for this fold
        meta_X_fold = np.hstack(fold_features)
        meta_y_fold = np.array(y_test, dtype=np.int64)
        all_features.append(meta_X_fold)
        all_labels.append(meta_y_fold)
        torch.cuda.empty_cache()

    # After all folds, align and stack features and labels
    if all_features:
        min_samples = min(f.shape[0] for f in all_features)
        all_features_aligned = [f[:min_samples] for f in all_features]
        all_labels_aligned = [l[:min_samples] for l in all_labels]
        meta_X = np.vstack(all_features_aligned)
        meta_y = np.concatenate(all_labels_aligned)

        # Scale features
        scaler = StandardScaler()
        meta_X = scaler.fit_transform(meta_X)
        joblib.dump(scaler, 'models_full/scaler.pkl')  # Save scaler

        # Fit the meta-model
        meta_model = LogisticRegression(max_iter=2000)
        meta_model.fit(meta_X, meta_y)
        joblib.dump(meta_model, 'models_full/meta_model.pkl')  # Save meta-model

        # Evaluate
        tta_probs = []
        for _ in range(3):
            probs = meta_model.predict_proba(meta_X)
            tta_probs.append(probs)
        final_probs = np.mean(tta_probs, axis=0)
        final_preds = np.argmax(final_probs, axis=1)
        print("\n=== Final Stacking Ensemble Performance ===")
        print(classification_report(meta_y, final_preds, target_names=classes, zero_division=0))

if __name__ == "__main__":
    main()

