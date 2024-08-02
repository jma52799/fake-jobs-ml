import os

def save_model_to_file(model, model_name, index, output_dir='../output/models/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Model file names
    json_file_path = os.path.join(output_dir, f'{model_name}_{index}.json')
    h5_file_path = os.path.join(output_dir, f'{model_name}_{index}.weights.h5')

    # Save model architecture as JSON
    model_json = model.to_json()
    with open(json_file_path, 'w') as json_file:
        json_file.write(model_json)
    
    # Save model weights as HDF5
    model.save_weights(h5_file_path)