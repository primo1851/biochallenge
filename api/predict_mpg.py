import pickle
import pandas as pd
import json

def predict_sound(model):
    
    # Make predictions using the loaded model
    predictions = model.predict(df)

    
    if y_pred == 'crying_baby':
        return 'Bebe chorando'
    elif y_pred == "doorbell":
        return 'Weak'
    elif y_pred == "siren":
        return 'Normal'
    elif y_pred == "speech":
        return 'fala'
  