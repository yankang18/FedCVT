from data_util.nus_wide_data_util import TwoPartyNusWideDataLoader


def get_data_with_multi_classes(data_dir, target_labels):
    print("target_label: {0}".format(target_labels))
    data_loader = TwoPartyNusWideDataLoader(data_dir)
    # image, text, labels = data_loader.get_train_data(target_labels=target_labels, binary_classification=False)
    image, text, labels = data_loader.get_test_data(target_labels=target_labels, binary_classification=False)
    print("image shape: {}".format(image.shape))
    print("text shape: {}".format(text.shape))
    print("labels shape: {}".format(labels.shape))


def get_data_with_binary_classes(data_dir, target_labels):
    data_loader = TwoPartyNusWideDataLoader(data_dir, binary_negative_label=0)
    # image, text, labels = data_loader.get_train_data(target_labels=target_labels, binary_classification=True)
    image, text, labels = data_loader.get_test_data(target_labels=target_labels, binary_classification=True)
    print("image shape: {}".format(image.shape))
    print("text shape: {}".format(text.shape))
    print("labels shape: {}".format(labels.shape))


if __name__ == "__main__":
    """
     top 10 labels:
     ['sky', 'clouds', 'person', 'water', 'animal',
     'grass', 'buildings', 'window', 'plants', 'lake']
    """
    # file_dir = "../../../Data/"
    file_dir = "/Users/yankang/Documents/Data/"

    #
    # Classification with 10 classes.
    #

    print("-" * 100)
    target_labels = ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']
    get_data_with_multi_classes(file_dir, target_labels)

    print("-" * 100)
    target_labels = None
    get_data_with_multi_classes(file_dir, target_labels)

    print("-" * 100)
    target_labels = ['sky', 'clouds']
    try:
        get_data_with_multi_classes(file_dir, target_labels)
    except Exception as exc:
        print(f"Exception occur:{exc}")

    print("-" * 100)
    target_labels = ["sky", "person"]
    get_data_with_binary_classes(file_dir, target_labels)

    print("-" * 100)
    target_labels = ["water"]
    get_data_with_binary_classes(file_dir, target_labels)

    print("-" * 100)
    target_labels = ["plants"]
    get_data_with_binary_classes(file_dir, target_labels)

    print("-" * 100)
    target_labels = ["sky", "person", "plants"]
    try:
        get_data_with_binary_classes(file_dir, target_labels)
    except Exception as exc:
        print(f"Exception occur:{exc}")
