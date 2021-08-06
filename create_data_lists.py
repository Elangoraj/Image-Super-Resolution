from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['Dataset/train/full_960x720'],
                      test_folders=['Dataset/test/England_960x720'],
                      min_size=100,
                      output_folder='Dataset/')
