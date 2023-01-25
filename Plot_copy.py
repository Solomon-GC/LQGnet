import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import log10
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 
        
# Style plot
# plt.style.use(['science','ieee', 'grid', 'no-latex'])
# Legend
Klegend = ["KNet - Train", "KNet - Validation", "KNet - Test", "Kalman Filter"]
loop_legend = ["KNet - Train", "KNet - Validation", "KNet - Test", "LQG", "LQR", "LQG - true system", "LQR - true system"]

# Color
KColor = ['-ro', 'k-', 'b-', 'g-', 'y']

class Plot:
    
    def __init__(self, pipeline):
        self.pipeline = pipeline


    def plot_epochs_simple(self, MSE_KF_true_system=None, fontSize=32, lineWidth=2, title=None, saveName=None, ylim=None, color=['-ro', 'k-', 'b-', 'g-', 'y-']):
        if title is None:
            title = self.pipeline.modelName

        # Figure
        plt.figure(figsize = (25, 10))

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        y_plt1 = self.pipeline.MSE_train_dB_epoch[x_plt]
        plt.plot(x_plt.cpu(), y_plt1.cpu(), color[0], label=Klegend[0], linewidth=lineWidth)

        y_plt2 = self.pipeline.MSE_val_dB_epoch[x_plt]
        plt.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=Klegend[1], linewidth=lineWidth)

        y_plt3 = self.pipeline.MSE_test_dB_avg_knet * torch.ones_like(x_plt)
        plt.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label=Klegend[2], linewidth=lineWidth)

        y_plt4 = self.pipeline.MSE_test_dB_avg_kf * torch.ones_like(x_plt)
        plt.plot(x_plt.cpu(), y_plt4.cpu(), color[3], label=Klegend[3], linewidth=lineWidth)

        if MSE_KF_true_system is not None:
            y_plt5 = MSE_KF_true_system * torch.ones_like(x_plt)
            plt.plot(x_plt.cpu(), y_plt5.cpu(), color[4], label='Kalman Filter - true system', linewidth=lineWidth)
        
        if ylim:
            plt.ylim(ylim)

        plt.legend(fontsize=fontSize)
        plt.xlabel('Epoch', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(title + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)


    def plot_epochs_simple_new_data(self, N_epochs, MSE_KF, MSE_train_epoch, MSE_val_epoch, MSE_test, options):
        ylim = options['ylim']
        color = options['color']
        lineWidth = options['linewidth']
        legend = options['legend']
        fontSize = options['fontsize']
        title = options['title']
        saveName = options['saveName']

        # Figure
        plt.figure(figsize = (25, 10))

        x_plt = torch.tensor(range(0, N_epochs))

        y_plt1 = MSE_train_epoch[x_plt]
        plt.plot(x_plt, y_plt1, color[0], label=legend[0], linewidth=lineWidth)

        y_plt2 = MSE_val_epoch[x_plt]
        plt.plot(x_plt, y_plt2, color[1], label=legend[1], linewidth=lineWidth)

        y_plt3 = MSE_test * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt3, color[2], label=legend[2], linewidth=lineWidth)

        y_plt4 = MSE_KF * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt4, color[3], label=legend[3], linewidth=lineWidth)
        
        if ylim:
            plt.ylim(ylim)

        plt.legend(fontsize=fontSize)
        plt.xlabel('Epoch', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(title + ": " + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)


    def plot_hist(self, MSE_KF_linear_arr, MSE_KNet_linear_arr, fontSize=32, title=None, saveName=None):
        plt.figure(figsize=(25, 10))
        
        sns.kdeplot(10*torch.log10(MSE_KF_linear_arr).cpu(), color='r', linewidth=3, label='Kalman Filter')
        sns.kdeplot(10*torch.log10(MSE_KNet_linear_arr).cpu(), color='g', linewidth=3, label='KalmanNet')
        
        plt.legend(fontsize=fontSize)
        plt.xlabel('MSE [dB]', fontsize=fontSize)
        
        if title is None:
            title = "Histogram [dB]"
        plt.title(title, fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)
    

    def plot_epochs_position_mse(self, MSE_KF, figSize=(25,8), fontSize=32, lineWidth=2, title=None, saveName=None, ylim=None, color=['k-', 'b-', 'g-', 'y-']):

        if title is None:
            title = self.pipeline.modelName

        # Figure
        plt.figure(figsize = figSize)

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # MSE validation
        y_plt1 = self.pipeline.MSE_val_position_dB_epoch[x_plt]
        plt.plot(x_plt.cpu(), y_plt1.cpu(), color[0], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_position_dB_avg_knet * torch.ones_like(x_plt)
        plt.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        plt.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        plt.plot(x_plt.cpu(), y_plt4.cpu(), color[3], label="Observation noise variance", linewidth=lineWidth)
        
        if ylim:
            plt.ylim(ylim)

        plt.legend(fontsize=fontSize)
        plt.xlabel('Epoch', fontsize=fontSize)
        plt.ylabel('Loss Value [dB]', fontsize=fontSize)
        plt.title(title + "Position MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)


    def plot_loop_combined(self, LQR_cost_dB, LQG_cost_KF_dB, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, color=['r-o','k-', 'b-', 'g-', 'y-']):
        
        if title is None:
            title = f"lr = {self.pipeline.learningRate}, weight decay = {self.pipeline.weightDecay}, batch size = {self.pipeline.N_B}"

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(2,1,1)
        ax2 = f.add_subplot(2,1,2)

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # LQR training loss each epoch
        y_plt1 = self.pipeline.LQR_train_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt1.cpu(), color[0], label=loop_legend[0], linewidth=lineWidth)

        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = LQG_cost_KF_dB * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt4.cpu(), color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = LQR_cost_dB * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt5.cpu(), color[4], label=loop_legend[4], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('LQR Loss Value [dB]', fontsize=fontSize)
        ax1.set_title("LQR Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax2.plot(x_plt.cpu(), y_plt1.cpu(), color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt2.cpu(), color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt3.cpu(), color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt4.cpu(), color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            f.savefig(self.pipeline.folderName + saveName)
    

    def plot_loop_lqr_and_mse(self, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(3,1,1) # Total loss = LQR + MSE
        ax2 = f.add_subplot(3,1,2) # LQR
        ax3 = f.add_subplot(3,1,3) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # Total training loss each epoch
        y_plt1 = self.pipeline.Total_loss_train_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt1.cpu(), color[0], label=loop_legend[0], linewidth=lineWidth)

        # Total validation loss each epoch
        y_plt2 = self.pipeline.Total_loss_val_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=loop_legend[1], linewidth=lineWidth)

        # Total test loss
        y_plt3 = self.pipeline.Total_loss_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label=loop_legend[2], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title(f"Total Loss: {self.pipeline.alpha}*MSE + {self.pipeline.beta}*LQR [dB] - per Epoch", fontsize=fontSize)


        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax2.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt4.cpu(), color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt5.cpu(), color[4], label=loop_legend[4], linewidth=lineWidth)

        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        # ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("LQR Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax3.plot(x_plt.cpu(), y_plt1.cpu(), color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax3.plot(x_plt.cpu(), y_plt2.cpu(), color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax3.plot(x_plt.cpu(), y_plt3.cpu(), color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax3.plot(x_plt.cpu(), y_plt4.cpu(), color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax3.set_ylim(ylim3)

        ax3.legend(fontsize=fontSize)
        ax3.set_xlabel('Epoch', fontsize=fontSize)
        ax3.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax3.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)

    def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
            rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

            pp = BboxPatch(rect, fill=False, **kwargs)
            parent_axes.add_patch(pp)

            p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
            inset_axes.add_patch(p1)
            p1.set_clip_on(False)
            p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
            inset_axes.add_patch(p2)
            p2.set_clip_on(False)

            return pp, p1, p2
        
    # LGQ Loss plot
    def plot_LQG_loss_vs_SNR(self, LQGNet_avg_loss_dB, MB_LQG_avg_loss_dB, LQGNet_MSE_loss_dB, MB_MSE_avg_loss_dB,
                       MM_LQGNet_avg_loss_dB, MM_MB_LQG_avg_loss_dB, MM_LQGNet_MSE_loss_dB, MM_MSE_avg_loss_dB, vdB):
        

        fontScale = 20
        msScale = 15
        lwScale = 3
        fig, ax = plt.subplots(figsize=(5,4))
        fig.tight_layout()
        ax.xaxis.set_tick_params(labelsize=fontScale)
        ax.yaxis.set_tick_params(labelsize=fontScale)
        ax.set_xticks(np.arange(-10,45,5), labels=['-10', '-5', '0', '5', '10','15','20','25','30','35','40'])
        [x.set_linewidth(2) for x in ax.spines.values()]
        # LQG
        plt.plot(vdB.cpu(), MB_LQG_avg_loss_dB[0,:].cpu(), 'o-', color='black', lw=lwScale, ms=msScale, label='KF+LQR w/o mismatch')
        plt.plot(vdB.cpu(), LQGNet_avg_loss_dB[0,:].cpu(), 'x--', color='red', lw=lwScale, ms=msScale, label='LQGNet w/o mismatch')

        plt.plot(vdB.cpu(), MM_MB_LQG_avg_loss_dB[0,:].cpu(), 's--', color='darkorange', markerfacecolor='none', lw=lwScale, ms=msScale, label='KF+LQR with mismatch')
        plt.plot(vdB.cpu(), MM_LQGNet_avg_loss_dB[0,:].cpu(), '*-', color='lightseagreen', markerfacecolor='none', lw=lwScale, ms=msScale, label='LQGNet with mismatch')
        # plt.plot(vdB.cpu(), 10**(MB_LQG_avg_loss_dB[0,:]/10).cpu(), 'o-', color='blue', lw=2, ms=5, label='KF+LQR')
        # plt.plot(vdB.cpu(), 10**(LQGNet_avg_loss_dB[0,:]/10).cpu(), 'x--', color='purple', lw=2, ms=5, label='LQGNet')
        # plt.yscale('log')
        plt.legend(loc='upper center', fancybox=True, shadow=True, fontsize=fontScale, ncol=2)
        plt.xlabel(r'$\frac{1}{r^2}$ [dB]', fontsize=fontScale)
        plt.ylabel('LGQ loss [dB]', fontsize=fontScale)
        plt.xlim(vdB.min().cpu(), vdB.max().cpu())
        plt.ylim(5,50)
        plt.grid(linestyle='--', linewidth=1)
        # fig.show()
        axins1 = ax.inset_axes([0.5, 0.3, 0.3, 0.3])
        # axins1 = zoomed_inset_axes(ax, zoom = 2, loc=10)

        for axis in ['top','bottom','left','right']:
            axins1.spines[axis].set_linewidth(2)
            axins1.spines[axis].set_color('black')
            
        axins1.plot(vdB.cpu(), MB_LQG_avg_loss_dB[0,:].cpu(), 'o-', color='black', lw=lwScale, ms=msScale)
        axins1.plot(vdB.cpu(), LQGNet_avg_loss_dB[0,:].cpu(), 'x--', color='red', lw=lwScale, ms=msScale)

        axins1.plot(vdB.cpu(), MM_MB_LQG_avg_loss_dB[0,:].cpu(), 's--', color='darkorange', markerfacecolor='none', lw=lwScale, ms=msScale)
        axins1.plot(vdB.cpu(), MM_LQGNet_avg_loss_dB[0,:].cpu(), '*-', color='lightseagreen', markerfacecolor='none', lw=lwScale, ms=msScale)
        axins1.set_xticks(np.arange(14,17,1), labels=['14','15','16'])

        # SPECIFY THE LIMITS
        x1, x2, y1, y2 = 14,16,8.5,10.5
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        # axins1.Axes.set_anchor( {'C'(15,15), 'SW', 'S', 'SE', 'E', 'NE','N', 'NW', 'W'}, share=False)
        # IF SET TO TRUE, TICKS ALONG 
        # THE TWO AXIS WILL BE VISIBLE
        # plt.xticks(visible=True)
        # plt.yticks(visible=True)

        
        def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
            rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

            pp = BboxPatch(rect, fill=False, **kwargs)
            parent_axes.add_patch(pp)

            p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
            inset_axes.add_patch(p1)
            p1.set_clip_on(False)
            p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
            inset_axes.add_patch(p2)
            p2.set_clip_on(False)

            return pp, p1, p2
        mark_inset(ax, axins1, loc1a=4, loc1b=1, loc2a=3, loc2b=2, fc="none", lw=2, ec="black")

        fig_path = 'Knet' + os.path.sep + 'LQGNet_performance_ICASSP.pdf'
        plt.savefig(fig_path, bbox_inches="tight")
        # fontScale = 20

        # fig, ax = plt.subplots(figsize=(10,6))
        # fig.tight_layout()
        # ax.xaxis.set_tick_params(labelsize=fontScale)
        # ax.yaxis.set_tick_params(labelsize=fontScale)
        # ax.set_xticks(np.arange(-10,45,5), labels=['-10', '-5', '0', '5', '10','15','20','25','30','35','40'])

        # # LQG
        # plt.plot(vdB.cpu(), MB_LQG_avg_loss_dB[0,:].cpu(), 'o-', color='black', lw=3, ms=20, label='KF+LQR w/o mismatch')
        # plt.plot(vdB.cpu(), LQGNet_avg_loss_dB[0,:].cpu(), 'x--', color='red', lw=3, ms=20, label='LQGNet w/o mismatch')

        # plt.plot(vdB.cpu(), MM_MB_LQG_avg_loss_dB[0,:].cpu(), 's--', color='darkorange', lw=3, ms=20, label='KF+LQR with mismatch')
        # plt.plot(vdB.cpu(), MM_LQGNet_avg_loss_dB[0,:].cpu(), '*-', color='lightseagreen', lw=3, ms=20, label='LQGNet with mismatch')
        # # plt.plot(vdB.cpu(), 10**(MB_LQG_avg_loss_dB[0,:]/10).cpu(), 'o-', color='blue', lw=2, ms=5, label='KF+LQR')
        # # plt.plot(vdB.cpu(), 10**(LQGNet_avg_loss_dB[0,:]/10).cpu(), 'x--', color='purple', lw=2, ms=5, label='LQGNet')
        # # plt.yscale('log')
        # plt.legend(loc='upper right', fontsize=fontScale, ncol=2)
        # plt.xlabel(r'$\frac{1}{r^2}$ [dB]', fontsize=fontScale)
        # plt.ylabel('LGQ loss [dB]', fontsize=fontScale)
        # plt.xlim(vdB.min().cpu(), vdB.max().cpu())
        # plt.grid(linestyle='--', linewidth=2)
        # fig.show()

        # fig_path = 'Knet' + os.path.sep + 'LQGNet_performance_ICASSP.pdf'
        # plt.savefig(fig_path, bbox_inches="tight")
        
        # # ------------- plot MSE ------------------
        # fig, ax = plt.subplots(figsize=(10,6))
        # fig.tight_layout()
        # ax.xaxis.set_tick_params(labelsize=fontScale)
        # ax.yaxis.set_tick_params(labelsize=fontScale)
        # ax.set_xticks(np.arange(-10,45,5), labels=['-10', '-5', '0', '5', '10','15','20','25','30','35','40'])

        # # LQG
        # plt.plot(vdB.cpu(), MB_MSE_avg_loss_dB[0,:].cpu(), 'o-', color='black', lw=3, ms=20, label='KF+LQR w/o mismatch')
        # plt.plot(vdB.cpu(), LQGNet_MSE_loss_dB[0,:].cpu(), 'x--', color='red', lw=3, ms=20, label='LQGNet w/o mismatch')

        # plt.plot(vdB.cpu(), MM_MSE_avg_loss_dB[0,:].cpu(), 's--', color='darkorange', lw=3, ms=20, label='KF+LQR with mismatch')
        # plt.plot(vdB.cpu(), MM_LQGNet_MSE_loss_dB[0,:].cpu(), '*-', color='lightseagreen', lw=3, ms=20, label='LQGNet with mismatch')
        # # plt.plot(vdB.cpu(), 10**(MB_LQG_avg_loss_dB[0,:]/10).cpu(), 'o-', color='blue', lw=2, ms=5, label='KF+LQR')
        # # plt.plot(vdB.cpu(), 10**(LQGNet_avg_loss_dB[0,:]/10).cpu(), 'x--', color='purple', lw=2, ms=5, label='LQGNet')
        # # plt.yscale('log')
        # plt.legend(loc='upper right', fontsize=fontScale, ncol=2)
        # plt.xlabel(r'$\frac{1}{r^2}$ [dB]', fontsize=fontScale)
        # plt.ylabel('MSE loss [dB]', fontsize=fontScale)
        # plt.xlim(vdB.min().cpu(), vdB.max().cpu())
        # plt.grid(linestyle='--', linewidth=2)
        # fig.show()

        # fig_path = 'Knet' + os.path.sep + 'LQGNet_performance_ICASSP.pdf'
        # plt.savefig(fig_path, bbox_inches="tight")
        
    def plot_lqr_and_mse(self, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(2,1,1) # LQR
        ax2 = f.add_subplot(2,1,2) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_epochs))

        # Total training loss each epoch
        y_plt1 = self.pipeline.Total_loss_train_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt1.cpu(), color[0], label=loop_legend[0], linewidth=lineWidth)

        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_ref_cost * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt4.cpu(), color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_ref_cost * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt5.cpu(), color[4], label=loop_legend[4], linewidth=lineWidth)

        if ylim2:
            ax1.set_ylim(ylim2)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title("LQG Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax2.plot(x_plt.cpu(), y_plt1.cpu(), color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt2.cpu(), color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt3.cpu(), color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        r2 = self.pipeline.ssModel.R.diag()[0]
        y_plt4 = 10 * log10(r2) * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt4.cpu(), color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax2.set_ylim(ylim3)

        ax2.legend(fontsize=fontSize)
        ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)


    def plot_control(self, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(3,1,1) # Control Loss
        ax2 = f.add_subplot(3,1,2) # LQR
        ax3 = f.add_subplot(3,1,3) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # training loss each epoch
        y_plt1 = self.pipeline.Control_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # validation loss each epoch
        y_plt2 = self.pipeline.Control_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # test loss TODO
        # y_plt3 = self.pipeline.Total_loss_test_dB_avg * torch.ones_like(x_plt)
        # ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title(f"Control Loss [dB] - per Epoch", fontsize=fontSize)


        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt5, color[4], label=loop_legend[4], linewidth=lineWidth)

        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        # ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("LQG Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax3.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt3, color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax3.set_ylim(ylim3)

        ax3.legend(fontsize=fontSize)
        ax3.set_xlabel('Epoch', fontsize=fontSize)
        ax3.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax3.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)
    

    def plot_loop_lqr_and_mse_model_mismatch(self, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-', 'g--', 'y--']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(3,1,1) # LQR + MSE
        ax2 = f.add_subplot(3,1,2) # LQR
        ax3 = f.add_subplot(3,1,3) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # Total training loss each epoch
        y_plt1 = self.pipeline.Total_loss_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # Total validation loss each epoch
        y_plt2 = self.pipeline.Total_loss_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # Total test loss
        y_plt3 = self.pipeline.Total_loss_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title(f"Total Loss: {self.pipeline.alpha}*MSE + {self.pipeline.beta}*LQR [dB] - per Epoch", fontsize=fontSize)


        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt5, color[4], label=loop_legend[4], linewidth=lineWidth)

        # LQG with true system
        y_plt6 = self.pipeline.LQG_cost_true_system * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt6, color[5], label=loop_legend[5], linewidth=lineWidth)

        # LQR with true system
        y_plt7 = self.pipeline.LQR_cost_true_system * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt7, color[6], label=loop_legend[6], linewidth=lineWidth)

        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        # ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("LQR Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax3.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt3, color[3], label="KF - True system", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax3.set_ylim(ylim3)

        ax3.legend(fontsize=fontSize)
        ax3.set_xlabel('Epoch', fontsize=fontSize)
        ax3.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax3.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)


    def plot_lqr_and_mse_model_mismatch(self, MSE_KF, LQG_correct_model,figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'r-', 'g-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(2,1,1) # LQR
        ax2 = f.add_subplot(2,1,2) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # Total training loss each epoch
        y_plt1 = self.pipeline.Total_loss_train_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt1.cpu(), color[0], label=loop_legend[0], linewidth=lineWidth)

        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt4.cpu(), color[3], label='LQG - wrong model', linewidth=lineWidth)

        # Correct model
        y_plt5 = LQG_correct_model * torch.ones_like(x_plt)
        ax1.plot(x_plt.cpu(), y_plt5.cpu(), color[4], label='LQG - correct model', linewidth=lineWidth)

        
        if ylim2:
            ax1.set_ylim(ylim2)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title("LQG Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax2.plot(x_plt.cpu(), y_plt1.cpu(), color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt2.cpu(), color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt3.cpu(), 'g-', label="KF - correct model", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt4.cpu(), 'y-', label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax2.set_ylim(ylim3)

        ax2.legend(fontsize=fontSize)
        ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)


    def plot_epochs_simple_model_mismatch(self, MSE_KF_true_system=None, fontSize=32, lineWidth=2, title=None, saveName=None, ylim=None, color=['-ro', 'k-', 'b-', 'r-', 'g-']):
        if title is None:
            title = self.pipeline.modelName

        # Figure
        plt.figure(figsize = (25, 10))

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        y_plt1 = self.pipeline.MSE_train_dB_epoch[x_plt]
        plt.plot(x_plt.cpu(), y_plt1.cpu(), color[0], label=Klegend[0], linewidth=lineWidth)

        y_plt2 = self.pipeline.MSE_val_dB_epoch[x_plt]
        plt.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=Klegend[1], linewidth=lineWidth)

        y_plt3 = self.pipeline.MSE_test_dB_avg_knet * torch.ones_like(x_plt)
        plt.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label=Klegend[2], linewidth=lineWidth)

        y_plt4 = self.pipeline.MSE_test_dB_avg_kf * torch.ones_like(x_plt)
        plt.plot(x_plt.cpu(), y_plt4.cpu(), color[3], label='KF - wrong model', linewidth=lineWidth)

        if MSE_KF_true_system is not None:
            y_plt5 = MSE_KF_true_system * torch.ones_like(x_plt)
            plt.plot(x_plt.cpu(), y_plt5.cpu(), color[4], label='KF - correct model', linewidth=lineWidth)
        
        if ylim:
            plt.ylim(ylim)

        plt.legend(fontsize=fontSize)
        plt.xlabel('Epoch', fontsize=fontSize)
        plt.ylabel('Loss Value [dB]', fontsize=fontSize)
        plt.title(title + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)


    def plot_control_model_mismatch(self, MSE_KF, LQG_correct, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(3,1,1) # Control Loss
        ax2 = f.add_subplot(3,1,2) # LQR
        ax3 = f.add_subplot(3,1,3) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # training loss each epoch
        y_plt1 = self.pipeline.Control_train_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt1.cpu(), color[0], label=loop_legend[0], linewidth=lineWidth)

        # validation loss each epoch
        y_plt2 = self.pipeline.Control_val_dB_epoch[x_plt]
        ax1.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=loop_legend[1], linewidth=lineWidth)

        # test loss TODO
        # y_plt3 = self.pipeline.Total_loss_test_dB_avg * torch.ones_like(x_plt)
        # ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title(f"Control Loss [dB] - per Epoch", fontsize=fontSize)


        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax2.plot(x_plt.cpu(), y_plt2.cpu(), color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt3.cpu(), color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt4.cpu(), 'r-', label='LQG - wrong model', linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = LQG_correct * torch.ones_like(x_plt)
        ax2.plot(x_plt.cpu(), y_plt5.cpu(), 'g-', label='LQG - correct model', linewidth=lineWidth)

        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        # ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("LQG Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax3.plot(x_plt.cpu(), y_plt1.cpu(), color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax3.plot(x_plt.cpu(), y_plt2.cpu(), color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax3.plot(x_plt.cpu(), y_plt3.cpu(), color[3], label="KF - correct model", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax3.plot(x_plt.cpu(), y_plt4.cpu(), color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax3.set_ylim(ylim3)

        ax3.legend(fontsize=fontSize)
        ax3.set_xlabel('Epoch', fontsize=fontSize)
        ax3.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax3.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)