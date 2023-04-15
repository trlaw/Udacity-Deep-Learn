import matplotlib.pyplot as plt
from IPython.display import display, clear_output

train_loss_title = 'Training Loss'
val_loss_title = 'Validation Loss'
plot_title_props = {'fontsize':16,'fontweight':'bold'}
    
# Dashboard Class
class TrainingDashboard:
    def __init__(self):
        self.fig, self.axs = plt.subplots(1, 2, \
                                          squeeze = True, \
                                          constrained_layout = True, \
                                          figsize=(12,5), \
                                          gridspec_kw={'wspace': 0.15, 'hspace': 0.08})
        
        self.train_loss_plot = self.axs[0]
        self.val_loss_plot = self.axs[1]
        self.train_loss_history = []
        self.val_loss_history = []
        
        plt.show()
    
    
    def repaint(self):
        # Command and allow redraw
        # https://medium.com/@shahinrostami/jupyter-notebook-and-updating-plots-f1ec4cdc354b
        clear_output(wait = True)
        display(self.fig)
        plt.pause(0.3)
    
    
    def paint_axes(self,ax):
        ax.set_xlabel('Epoch')
        ax.set_ylim(0,4)
    
    
    def update_training(self,training_loss):
        
        # Update trace data
        self.train_loss_history.append(training_loss)
        
        # Indicate latest and extrema in titles
        self.train_loss_plot.set_title( \
                            f'{train_loss_title} (Last: {training_loss:.3f}, Min: {min(self.train_loss_history):.3f})', \
                            plot_title_props, pad = 10 \
                                      )
        
        # Update plot
        self.paint_axes(self.train_loss_plot)
        self.train_loss_plot.plot(self.train_loss_history)
        self.repaint()

        
    def update_validation(self,validation_loss):
        # Update trace data
        self.val_loss_history.append(validation_loss)
        
        # Indicate latest and extrema in titles
        self.val_loss_plot.set_title( \
                                f'{val_loss_title} (Last: {validation_loss:.3f}, Min: {min(self.val_loss_history):.3f})', \
                                plot_title_props, pad = 10 \
                                    )
        
        # Update plots
        self.paint_axes(self.val_loss_plot)
        self.val_loss_plot.plot(self.val_loss_history)
        self.repaint()