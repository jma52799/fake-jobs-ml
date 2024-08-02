from keras.models import model_from_json
import sys


def load_model(model_name, index):
    try:
        with open(f'../output/models/{model_name}_{index}.json', 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(f'../output/models/{model_name}_{index}.h5')
        return model
    except FileNotFoundError as e: 
        print(f"Error: {e}")
        print(f"One or both of the model or model weights files do not exist.")
        print("Please make sure you've train the model first and try again.")
        sys.exit(1)


