import matplotlib.pyplot as plt



def bar_plot(acc_dict):
    plt.figure(figsize=(10, 5))
    plt.bar(acc_dict.keys(), acc_dict.values(), color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy by Year-to-Year Setting")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




def draw_learning_curve(model):
    results = model.evals_result()

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train Log Loss')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Validation Log Loss')
    plt.xlabel('Boosting Rounds', fontsize=18)
    plt.ylabel('Log Loss', fontsize=18)
    #plt.title(f'Learning Curve', fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()