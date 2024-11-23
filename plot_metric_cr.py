from matplotlib import pyplot as plt

if __name__ == '__main__':
    cr = [13,50,200]
    MSEs = [0.0087, 0.0162, 0.0280]
    NMSEs = [0.0221, 0.0424, 0.0740]
    PSNRs = [21.5310, 18.4136, 15.9943]
    SSIMs = [0.9987, 0.9976, 0.9958]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
    axs[0][0].plot(cr, MSEs, label='MSE', marker='o')
    axs[0][0].title.set_text('MSE')
    axs[0][1].plot(cr, NMSEs, label='NMSE', marker='o')
    axs[0][1].title.set_text('NMSE')
    axs[1][0].plot(cr, PSNRs, label='PSNR', marker='o')
    axs[1][0].title.set_text('PSNR')
    axs[1][1].plot(cr, SSIMs, label='SSIM', marker='o')
    axs[1][1].title.set_text('SSIM')
    plt.tight_layout()
    plt.savefig('./outputs_before/metrics_cr.png')
