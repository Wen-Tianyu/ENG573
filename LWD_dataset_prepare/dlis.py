from dlisio import dlis
import numpy as np
from matplotlib import pyplot as plt

with dlis.load('./data/slb_data_4_sec_av_s1.dlis')[0] as file:
    for frame in file.frames:
        i = 0
        for channel in frame.channels:
            if 'S1' in channel.name:
                data = channel.curves()
                print(data.shape)
                data = np.nan_to_num(data, nan=0.)

                plt.figure(figsize=(1, 100))
                plt.imshow(data, cmap='YlOrBr')
                plt.axis('off')
                plt.savefig('./{}.png'.format(i))

                i += 1
