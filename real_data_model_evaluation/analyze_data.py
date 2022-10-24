import pandas as pd
import argparse, glob

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--Path")
parser.add_argument("-acc", "--Accuracy", action='store_true')
parser.add_argument("-lat", "--Latency", action='store_true')
args = parser.parse_args()

path_to_analyze = args.Path
analyze_accuracy = bool(args.Accuracy)
analyze_latency = bool(args.Latency)

print("Number of files to analyze: {}".format(len(glob.glob(path_to_analyze + "/*/*.txt"))))

csv_list = []
labels_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
               "L", "M", "N", "Nothing", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

print(len(labels_list))

if analyze_latency:
    for file in glob.glob(path_to_analyze + "/*/*.txt"):
        data = pd.read_csv(file, sep=';', names=['Frame', 'Gesture', 'Probability', 'Latency'])
        csv_list.append(data)

    if len(csv_list) > 1:
        csv_merged = pd.concat(csv_list, ignore_index=True)
    else:
        csv_merged = csv_list[0]
    sorted_csv = csv_merged.sort_values(by=['Frame'], ascending=True)

    print(100 * '-')
    print("Inference summary: ")
    print(100 * '-')
    print("Min latency [ms]: {}".format(sorted_csv['Latency'].min()))
    print("Max latency [ms]: {}".format(sorted_csv['Latency'].max()))
    print("Average latency [ms]: {}".format(sorted_csv['Latency'].mean()))
    print("Median of latency [ms]: {}".format(sorted_csv['Latency'].median()))
    print("Standard deviation of latency [ms]: {}".format(sorted_csv['Latency'].std()))
    print(100 * '-')

    print("Frame:")
    for line in sorted_csv['Frame'].tolist():
        print(str(line))
    print("Latency:")
    for line in sorted_csv['Latency'].tolist():
        print(str(line))

elif analyze_accuracy:
    for label in labels_list:
        for file in glob.glob(path_to_analyze + "/" + str(label) + "/*.txt"):
            csv_data = pd.read_csv(file, sep=';', names=['Frame', 'Gesture', 'Probability', 'Latency'])
            sorted_csv = csv_data.sort_values(by=['Frame'], ascending=True)

        correct_predictions = sorted_csv[sorted_csv['Gesture'].astype(str).str.contains(label)]
        num_of_predictions = sorted_csv.shape[0]
        num_of_correct_predictions = correct_predictions.shape[0]
        most_frequent_recognized_class = str(sorted_csv["Gesture"].value_counts()[:1].index.tolist()[0])
        print("Number of predictions (frames): {}".format(num_of_predictions))
        print("Most frequent recognized class '{}'".format(most_frequent_recognized_class.strip()))
        print("Number of correct predictions for gesture '{}': {}".format(label, num_of_correct_predictions))
        print("Percentage of correct predictions for gesture '{}': {}".format(label,
                                                                              num_of_correct_predictions / num_of_predictions))
        print(100 * '-')

else:
    print("Please select type of analysis: latency ('-lat') or accuracy ('-acc')")
