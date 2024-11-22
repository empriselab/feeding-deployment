"""
Parse result files and find mean and standard deviation of the results
"""

import os

def parse_results(result_path):
    
    with open(result_path, 'r') as f:
        lines = f.readlines()

    threshold, accuracy = None, None

    # find the threshold (after "Best Threshold: ")
    for line in lines:
        if "Best Threshold: " in line:
            threshold = float(line.split("Best Threshold: ")[1])
        if "Best Accuracy: " in line:
            accuracy = float(line.split("Best Accuracy: ")[1])
    
    if threshold is None:
        threshold = 0.0
    if accuracy is None:
        print(f"Error: No accuracy found in {result_path}")
        accuracy = 0.0
    
    return threshold, accuracy
    

def parse_dataset(source_path):
    
    accuracys = []
    for i in range(10):
        result_path = source_path + f'/results/{i}.txt'
        threshold, accuracy = parse_results(result_path)
        accuracys.append(accuracy)
    
    mean_accuracy = sum(accuracys) / len(accuracys)
    std_accuracy = (sum([(accuracy - mean_accuracy)**2 for accuracy in accuracys]) / len(accuracys))**0.5

    return mean_accuracy, std_accuracy

if __name__ == "__main__":
    source_paths = ['blinking', 'eyebrows_raised', 'head_nod', 'head_still_atleast_three_secs', 'look_at_robot_atleast_three_secs', 'talking']
    
    means = []
    stds = []
    for source_path in source_paths:
        mean, std = parse_dataset( 'gesture_data/' + source_path)
        print(f"Gestures: {source_path}, Mean: {mean}, Std: {std}")
        means.append(mean)
        stds.append(std)

    print(f"Overall Mean: {sum(means)/len(means)}, Std of Means: {(sum([(mean - sum(means)/len(means))**2 for mean in means]) / len(means))**0.5}")