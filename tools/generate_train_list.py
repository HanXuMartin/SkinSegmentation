import os
import random


def generate_train_list(data_set_txt: str,
                        val_split: float = 0.1) -> None:
    with open(data_set_txt, "r") as data_list_file:
        data_list = data_list_file.readlines()
    data_length = len(data_list)
    assert data_length > 0, "No images found"
    val_lenth  = int(data_length * val_split)
    random.shuffle(data_list)
    train_data_txt = r"./docs/train_list.txt"
    val_data_txt = r"./docs/val_list.txt"
    with open(train_data_txt, "w") as train_data_file:
        for image_path in data_list[:-val_lenth]:
            train_data_file.write(image_path)
    with open(val_data_txt, "w") as val_data_file:
        for image_path in data_list[-val_lenth:]:
            val_data_file.write(image_path)


def generate_data_list() -> None:
    ecu_data_prefix = r"E:\DeepLearning\Segmentation\ECU"
    image_dir = os.path.join(ecu_data_prefix, "images")
    data_set_path = r"./docs/data_list.txt"
    with open(data_set_path, "w") as data_list:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if not file.endswith(".jpg"): continue
                image_path = os.path.join(root, file)
                # image_path = image_path.replace(ecu_data_prefix, "")
                data_list.write(f"{image_path}\n")
    return data_set_path
    

def main() -> None:
    data_set_path = generate_data_list()
    generate_train_list(data_set_path)
    print("Finish")

if __name__ == "__main__":
    main()