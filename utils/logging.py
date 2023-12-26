import csv
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

class LoggingStrategy:
    def tensorboard_log(training_loss, testing_loss, metric_scores, epoch, log_dir='logs'):
        writer = SummaryWriter(log_dir)

        writer.add_scalar('Loss/Train', training_loss, epoch)
        writer.add_scalar('Loss/Test', testing_loss, epoch)

        for metric, score in metric_scores.items():
            writer.add_scalar(f'Metric/{metric}', score, epoch)

        writer.close()

    def csv_log(training_loss, testing_loss, metric_scores, epoch, file_name='log.csv'):
        fieldnames = ['epoch', 'training_loss', 'testing_loss'] + list(metric_scores.keys())
        row = {'epoch': epoch, 'training_loss': training_loss, 'testing_loss': testing_loss, **metric_scores}

        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:  
                writer.writeheader()
            writer.writerow(row)

    def excel_log(training_loss, testing_loss, metric_scores, epoch, file_name='log.xlsx'):
        row = {'Epoch': epoch, 'Training Loss': training_loss, 'Testing Loss': testing_loss, **metric_scores}
        df = pd.DataFrame([row])

        with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            if not writer.sheets:
                df.to_excel(writer, index=False)
            else:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)