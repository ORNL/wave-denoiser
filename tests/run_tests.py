import h5py
import numpy as np
from wavedenoiser.wavedenoiser import Denoiser
import matplotlib.pyplot as plt
#
if __name__ == "__main__":
    model = Denoiser()
    #
    fid = h5py.File("example_waveforms.h5", "r")
    data_group = fid["data"]
    example_data = []
    for akey in data_group.keys():
        dataset = data_group[akey]
        example_data.append(dataset[...])
    fid.close()
    #
    example_data = np.array(example_data)
    preds = model.predict(example_data)
    #
    sampling_rate = 100
    chan_name_list = ['East', 'North', 'Vertical']
    for isample in range(len(example_data)):
        data_per_station = example_data[isample]
        pred_per_station = preds[isample].detach().numpy()
        fig, axs = plt.subplots(6, 1, sharex=True, sharey=True, layout='tight', figsize=(8,12))
        for ichan in range(len(data_per_station)):
            ax = axs[ichan*2]
            time = np.arange(len(data_per_station[ichan]))/sampling_rate
            ax.plot(time, data_per_station[ichan], 'k-', lw=0.5, label=chan_name_list[ichan])
            #
            ax.set_xlim(time[0], time[-1])
            ax.set_ylabel('Amp (count)')
            ax.legend(ncols=1, fontsize=10)
            ax = axs[ichan*2+1]
            time = np.arange(len(pred_per_station[ichan]))/sampling_rate
            ax.plot(time, pred_per_station[ichan], 'r-', lw=0.5, label=f'{chan_name_list[ichan]} Denoised')
            #
            ax.set_xlim(time[0], time[-1])
            ax.set_ylabel('Amp (count)')
            ax.legend(ncols=1, fontsize=10)
        #
        axs[-1].set_xlabel('Time (s)')
        fig.savefig(f'example_waveform_{isample}.jpg', dpi=300)
        plt.close()

