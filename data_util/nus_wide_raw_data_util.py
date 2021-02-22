import csv
from io import BytesIO

import numpy as np
import requests
from PIL import Image


def get_images():
    image_urls = "D:/Data/NUS_WIDE/NUS_WIDE/NUS-WIDE-urls/NUS-WIDE-urls.txt"
    # df = pd.read_csv(image_urls, header=0, sep=" ")
    # print(df.head(10))
    # kkk = df.loc[:, "url_Middle"]
    # print(kkk.head(10))

    read_num_urls = 1
    with open(image_urls, "r") as fi:
        fi.readline()
        reader = csv.reader(fi, delimiter=' ', skipinitialspace=True)
        for idx, row in enumerate(reader):
            if idx >= read_num_urls:
                break
            print(row[0], row[2], row[3], row[4])
            if row[3] is not None and row[3] != "null":
                url = row[4]
                print("{0} url: {1}".format(idx, url))

                str_array = row[0].split("\\")
                print(str_array[3], str_array[4])

                # img = imageio.imread(url)
                # print(type(img), img.shape)

                response = requests.get(url)
                print(response.status_code)
                img = Image.open(BytesIO(response.content))
                arr = np.array(img)
                print(type(img), arr.shape)
                # imageio.imwrite("", img)
                size = 48, 48
                img.thumbnail(size)
                img.show()
                arr = np.array(img)
                print("thumbnail", arr.shape)


if __name__ == '__main__':
    get_images()
