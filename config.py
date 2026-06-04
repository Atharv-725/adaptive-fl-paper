class Config:
    NUM_CLIENTS = 10
    NUM_ROUNDS = 30
    FRACTION_FIT = 1.0
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    INPUT_SIZE = 784
    HIDDEN_SIZE = 128
    NUM_CLASSES = 10
    DATA_DIR = './data'
    IID = True
    DRIFT_CLIENTS = 3
    DRIFT_START_ROUND = 10
    RESULTS_DIR = './results'
    PLOTS_DIR = './results/plots'
    CSV_DIR = './results/csv'
