import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check class distribution
train_dir = 'dataset/archive/chest_xray/train'
print("Normal:", len(os.listdir(f'{train_dir}/NORMAL')))
print("Pneumonia:", len(os.listdir(f'{train_dir}/PNEUMONIA')))

# Data augmentation & normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    shear_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'dataset/archive/chest_xray/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    'dataset/archive/chest_xray/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
#build model
#transfer learning
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

#compile model
from tensorflow.keras.optimizers import Adam


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']

)

model.summary()
#handles classimbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)
#train model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_checkpoint = ModelCheckpoint(
    'best_pneumonia_model.h5',
    monitor='val_loss',
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Skip first fit phase since weights are already saved
# history = model.fit(
#     train_gen,
#     epochs=20,
#     validation_data=val_gen,
#     class_weight=class_weight_dict,
#     callbacks=[early_stopping, model_checkpoint, reduce_lr]
# )
#evaluate model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Load best model
model.load_weights('best_pneumonia_model.h5')

# Predict on validation set
val_pred_probs = model.predict(val_gen).flatten()
val_pred_labels = (val_pred_probs > 0.5).astype(int)

# Classification report
print("Classification Report:")
print(classification_report(val_gen.classes, val_pred_labels, target_names=['NORMAL', 'PNEUMONIA']))

# Confusion matrix
cm = confusion_matrix(val_gen.classes, val_pred_labels)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(val_gen.classes, val_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
# #plot training history (skipped since `history` is undefined)
# plt.figure(figsize=(12, 5))
# 
# # Accuracy plot
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='lower right')
# 
# # Loss plot
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper right')
# 
# plt.tight_layout()
# plt.show()
#fine-tuning
base_model.trainable = True

# Freeze all layers except the last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Re-compile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

# Continue training fine-tuned model
history_ft = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

#evaluate fine-tuned model
# Predict on validation set
val_pred_probs_ft = model.predict(val_gen).flatten()
val_pred_labels_ft = (val_pred_probs_ft > 0.5).astype(int)

# Classification report
print("\nFine-Tuned Model Classification Report:")
print(classification_report(val_gen.classes, val_pred_labels_ft, target_names=['NORMAL', 'PNEUMONIA']))

# Confusion matrix
cm_ft = confusion_matrix(val_gen.classes, val_pred_labels_ft)
plt.figure(figsize=(6,6))
sns.heatmap(cm_ft, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Fine-Tuned Model Confusion Matrix')
plt.show()

#plot fine-tuning history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history_ft.history['accuracy'], label='Train Accuracy')
plt.plot(history_ft.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-Tuned Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history_ft.history['loss'], label='Train Loss')
plt.plot(history_ft.history['val_loss'], label='Validation Loss')
plt.title('Fine-Tuned Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
