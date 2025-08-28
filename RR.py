from PySide6.QtCore import Qt, QSettings, QAbstractTableModel
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTableView,
    QCheckBox,
    QInputDialog,
    QFileDialog,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialogButtonBox,
    QMessageBox,
)
from PySide6 import QtGui, QtCore

import sys
from pycomm3 import LogixDriver
import csv
import pandas as pd
import re
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tabulate import tabulate
import matplotlib.pyplot as plt
import qdarktheme
import pandas as pd

def run_CGK(file_name, selected_cgk=None):
    print(f'File {file_name}')
    df = pd.read_csv(file_name)

    # Require user-selected CGK measurements
    if selected_cgk is None or len(selected_cgk) == 0:
        raise ValueError('No CGK measurements selected. Please use Select CGK to choose columns, tolerances, and known values.')

    # selected_cgk: list of tuples (column_name, tolerance_value, known_value)
    measurements = [m for m, _, _ in selected_cgk]
    tolerances = [t for _, t, _ in selected_cgk]
    known_values = [k for _, _, k in selected_cgk]

    # Ensure numeric and absolute values
    for measurement in measurements:
        df[measurement] = pd.to_numeric(df[measurement], errors='coerce').abs()

    results = [['Measurement', 'CGK']]

    for i, measurement in enumerate(measurements):
        SV = df[measurement].std() * 3
        tol_10 = tolerances[i] * .1
        ave = df[measurement].mean()
        if SV == 0 or np.isnan(SV):
            CGK = np.nan
        else:
            CGK = (tol_10 - abs(ave - known_values[i])) / SV
        print(f'{measurement} std: {SV}, mean: {ave}, tol10: {tol_10}, cgk: {CGK}, known: {known_values[i]}')
        results.append([measurement, CGK])

    return pd.DataFrame(results[1:], columns=results[0])


def run_RR(csv_name, boxplots=False, scatterplots=False, type1=False, show_part_data=False, selected_measurements=None):

    # Load the data from CSV
    df = pd.read_csv(csv_name)

    # Ensure Part and Nest columns exist
    if 'Part' not in df.columns:
        df['Part'] = 1
    if 'Nest' not in df.columns:
        df['Nest'] = 1

    #df['ModelName'] = df['Nest']

    # Require user-selected measurements; no defaults
    if selected_measurements is None or len(selected_measurements) == 0:
        raise ValueError('No measurements selected. Please use Select Measurements to choose at least one column and tolerance.')

    # selected_measurements is a list of tuples: (column_name, tolerance_value)
    measurements = [m for m, _ in selected_measurements]
    tolarances = [t for _, t in selected_measurements]
    # Ensure numeric columns
    for measurement in measurements:
        df[measurement] = pd.to_numeric(df[measurement], errors='coerce')

    # absolute value each measurement
    for measurement in measurements:
        df[measurement] = df[measurement].abs()
    
    if type1:
        # set part and nest columns to all be 1
        df['Part'] = 1
        df['Nest'] = 1
        per_part_outputs = ['CG', 'Range', 'Min', 'Max']
    else:
        per_part_outputs = ['Range', 'Min', 'Max']

    output_types = ['', ' * 6', '% of Tolarance']
    outputs = []
    # Always show GRR summary only
    types = ['GRR Variation']
    outputs.extend([f'{type} {output_type}' for type in types for output_type in output_types])

    if not type1:
        # Summarized part stats headers instead of listing each part
        if show_part_data:
            outputs.append('Max Part Range')
            outputs.append('Average Part Range')
            outputs.append('Part Range StDev')
    else:
        if show_part_data:
            for output in per_part_outputs:
                outputs.append(f'{output}')

    temp = ['Measurement'] + outputs + ['Result']

    results = [temp]


    if boxplots:
        # dynamic grid based on number of measurements
        num_plots = len(measurements)
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = int(np.ceil(num_plots / rows))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
        fig.suptitle('Measurement Boxplots')

        # Flatten axes for easy indexing
        axes_list = np.array(axs).reshape(-1) if isinstance(axs, (list, np.ndarray)) else np.array([axs])

        for i, measurement in enumerate(measurements):
            ax = axes_list[i]
            df.boxplot(column=measurement, by='Part', ax=ax, patch_artist=True)
            ax.set_title(measurement)
            ax.set_xlabel('Part')
            ax.set_ylabel(measurement)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
            # set y axis to 1.5 times the range of the data
            ax.set_ylim(df[measurement].min() - (df[measurement].max() - df[measurement].min()) / 2, df[measurement].max() + (df[measurement].max() - df[measurement].min()) / 2)

        # Hide any unused axes
        for j in range(num_plots, len(axes_list)):
            axes_list[j].axis('off')

        # space out subplots more
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.5)

        # set plot window title
        fig.canvas.manager.set_window_title('Measurement Box Plots')

        # set window size roughly proportional to grid
        fig.set_size_inches(max(10, cols * 5), max(8, rows * 4))

        plt.show()

    if scatterplots:
        # dynamic grid based on number of measurements
        num_plots = len(measurements)
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = int(np.ceil(num_plots / rows))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
        fig.suptitle('Measurement Scatter Plots')

        # Flatten axes for easy indexing
        axes_list = np.array(axs).reshape(-1) if isinstance(axs, (list, np.ndarray)) else np.array([axs])

        # line plot with dots only
        for i, measurement in enumerate(measurements):
            ax = axes_list[i]
            df.plot.scatter(x='Part', y=measurement, ax=ax)
            ax.set_title(measurement)
            ax.set_xlabel('Part')
            ax.set_ylabel(measurement)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
            # set y axis to 1.5 times the range of the data
            ax.set_ylim(df[measurement].min() - (df[measurement].max() - df[measurement].min()) / 4, df[measurement].max() + (df[measurement].max() - df[measurement].min()) / 4)
            # Make the points smaller (overlay)
            ax.scatter(df['Part'], df[measurement], s=10)

        # Hide any unused axes
        for j in range(num_plots, len(axes_list)):
            axes_list[j].axis('off')

        # space out subplots more
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.5)

        # set plot window title
        fig.canvas.manager.set_window_title('Measurement Scatter Plots')

        # set window size roughly proportional to grid
        fig.set_size_inches(max(10, cols * 5), max(8, rows * 4))

        plt.show()

    for i, measurement in enumerate(measurements):

        # Fit the ANOVA model
        model = ols(f'{measurement} ~ C(Part) + C(Nest) + C(Part):C(Nest)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=1)
        
        # Sum of Squares
        SS_Total = np.sum((df[measurement] - df[measurement].mean())**2)
        SS_Part = anova_table.loc['C(Part)', 'sum_sq']
        SS_Nest = anova_table.loc['C(Nest)', 'sum_sq']
        SS_Part_Nest = anova_table.loc['C(Part):C(Nest)', 'sum_sq']
        SS_Repeat = anova_table.loc['Residual', 'sum_sq']

        # Degrees of Freedom
        DF_Total = df.shape[0] - 1
        DF_Part = len(df['Part'].unique()) - 1
        DF_Nest = len(df['Nest'].unique()) - 1
        DF_Part_Nest = DF_Part * DF_Nest
        print(DF_Part)
        print(DF_Nest)
        print(DF_Part_Nest)
        DF_Repeat = DF_Total - (DF_Part + DF_Nest + DF_Part_Nest)

        # Mean Squares
        MS_Part = SS_Part / DF_Part
        MS_Nest = SS_Nest / DF_Nest
        MS_Part_Nest = SS_Part_Nest / DF_Part_Nest
        MS_Repeat = SS_Repeat / DF_Repeat

        # Variance Components
        Var_Part = (MS_Part - MS_Part_Nest) / (len(df['Nest'].unique()) * 3) if MS_Part > MS_Part_Nest else 0
        Var_Nest = (MS_Nest - MS_Part_Nest) / (len(df['Part'].unique()) * 3) if MS_Nest > MS_Part_Nest else 0
        Var_Part_Nest = (MS_Part_Nest - MS_Repeat) / 3 if MS_Part_Nest > MS_Repeat else 0
        Var_Repeat = MS_Repeat

        # Study Variation
        Study_Variation = np.sqrt(Var_Part + Var_Nest + Var_Part_Nest + Var_Repeat)
        GRR_Variation = np.sqrt(Var_Nest + Var_Part_Nest + Var_Repeat)
        Repeatability = np.sqrt(Var_Repeat)
        Reporducibility = np.sqrt(Var_Nest + Var_Part_Nest)
        
        # Percent of Tolerance
        Tolerance = tolarances[i] # Replace with actual tolerance value
        Total_Percent_Tolerance = f'{((Study_Variation * 6 / Tolerance) * 100):.2f}%'
        GRR_Percent_Tolerance = f'{((GRR_Variation * 6 / Tolerance) * 100):.2f}%'
        GRR_Percent_Tolerance_num = ((GRR_Variation * 6 / Tolerance) * 100)
        Repeatability_Percent_Tolerance = f'{((Repeatability * 6 / Tolerance) * 100):.2f}%'
        Reporducibility_Percent_Tolerance = f'{((Reporducibility * 6 / Tolerance) * 100):.2f}%'

        part_stats = []
        if show_part_data:
            if type1:
                value_range = f'{df[measurement].max() - df[measurement].min():.4f}mm'
                min_value = f'{df[measurement].min():.4f}mm'
                max_value = f'{df[measurement].max():.4f}mm'
                stdev = df[measurement].std() * 3

                CG = (Tolerance*.1)/stdev

                part_stats.extend([CG, value_range, min_value, max_value])
            else:
                # Compute per-part ranges, then summarize
                part_ranges = []
                for part in df['Part'].unique():
                    part_df = df[df['Part'] == part]
                    part_ranges.append((part_df[measurement].max() - part_df[measurement].min()))

                if len(part_ranges) > 0:
                    max_range = np.max(part_ranges)
                    avg_range = np.mean(part_ranges)
                    std_range = np.std(part_ranges, ddof=1) if len(part_ranges) > 1 else 0.0
                    part_stats.extend([f'{max_range:.4f}mm', f'{avg_range:.4f}mm', f'{std_range:.4f}mm'])


        # Check if Study Variation is within 10% of Tolerance
        if GRR_Percent_Tolerance_num <= 10:
            result = "Acceptable"
        elif GRR_Percent_Tolerance_num <= 12:
            result = "Close As Hell"
        elif GRR_Percent_Tolerance_num <= 15:
            result = "Close"
        elif GRR_Percent_Tolerance_num >= 25:
            result = "Is Ass"
        else:
            result = "NOT Acceptable (15% - 25%)"

        output = []

        output = [measurement, f'{GRR_Variation:.6f}', f'{GRR_Variation * 6:.6f}', GRR_Percent_Tolerance]

        output.extend(part_stats)

        output.append(result)
        
        results.append(output)

    #print(tabulate(results, headers='firstrow', tablefmt='fancy_grid'))
    return pd.DataFrame(results[1:], columns=results[0])

def format_csv(name, file):

    def extract_index(tag):
        match = re.search(r'\[(\d+)\]', tag)
        return int(match.group(1)) if match else None

    def extract_child_names(tag):
        match = re.search(r'\]\.(.+)', tag)
        return match.group(1) if match else None

    df = pd.read_csv(f'{file}_raw.csv')

    df['Index'] = df['tag'].apply(extract_index)
    df['child_name'] = df['tag'].apply(extract_child_names)

    df_pivot = df.pivot_table(index='Index', columns='child_name', values='value', aggfunc='first')

    df_pivot = df_pivot.rename(columns = {'PartNumber':'Nest'})
    df_pivot['Part'] = ''

    df_pivot.reset_index(inplace=True)

    df_pivot.to_csv(f'{file}.csv', index=False)

def crawl_and_format(obj, name, data, start_index=0):
    """
    Recursively crawls through a dictionary or list and formats the data into a flattened dictionary.

    Args:
        obj (dict or list): The object to crawl through.
        name (str): The name of the object.
        data (dict): The flattened dictionary to store the formatted data.

    Returns:
        dict: The flattened dictionary with the formatted data.
    """
    # obj is a dictionary
    if isinstance(obj, dict):
        for key, value in obj.items():
            data = crawl_and_format(value, f'{name}.{key}', data)
    # obj is a list
    elif isinstance(obj, list):
        # iterate through the list
        for i, value in enumerate(obj):
            data = crawl_and_format(value, f'{name}[{i + start_index}]', data)
    # obj is an elementary object
    else:
        data[f'{name}'] = f'{obj}'

    return data


def data_to_dict(data):

    processed_data = []

    if isinstance(data, list):
        for tag in data:
            if isinstance(tag.value, list):
                for i, value in enumerate(tag.value):
                    processed_data.append({f'{tag.tag}[{str(i)}]': value})
            else:
                processed_data.append({tag.tag: tag.value})
    else:
        processed_data.append({data.tag: data.value})

    return processed_data

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(
                        item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    return dict(items)

def write_to_csv(data, csv_file, csv_file_input):
    with open(f'{csv_file_input}_raw.csv', 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['tag', 'value'])
        writer.writeheader()
        for item in data:
            for tag, value in item.items():
                writer.writerow({'tag': tag, 'value': value})

    format_csv(csv_file, csv_file_input)

def read_tag(tag_input, ip_input, file_name_input):
        if ip_input == '':
            ip = '192.137.81.51'
        else:
            ip = ip_input

        if tag_input == '':
            tag = 'Cell_PartDataHistory{2500}'
        else:
            tag = tag_input

        with LogixDriver(ip) as plc:
            read_result = plc.read(tag)
        
        print(read_result)
        
        data = data_to_dict(read_result)
        data = [flatten_dict(item) for item in data]
        write_to_csv(data, tag, file_name_input)

class TableWindow(QMainWindow):
    def __init__(self, parent, data):
        super(TableWindow, self).__init__(parent)

        self.data = data
        
        # get the first row as headers in the dataframe and then remove it
        headers = data.iloc[0]
        data = data[1:]

        self.model = TableModel(data, headers)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.setCentralWidget(self.table)
        
        # Resize columns and rows to fit the content
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()


        # Calculate the required size
        width = self.table.verticalHeader().width() + self.table.horizontalHeader().length() + self.table.frameWidth() * 2
        height = self.table.horizontalHeader().height() + self.table.verticalHeader().length() + self.table.frameWidth() * 2

        # Set the window size
        self.resize(width, height)

        # add the scrollbar size to the window size if needed
        if self.table.verticalScrollBar().isVisible():
            width += self.table.verticalScrollBar().width()
        if self.table.horizontalScrollBar().isVisible():
            height += self.table.horizontalScrollBar().height()

        # Set the window size with the scrollbar
        self.resize(width, height)
    


class TableModel(QAbstractTableModel):

    def __init__(self, data, headers):
        super(TableModel, self).__init__()
        self._data = data
        self._headers = headers


    def data(self, index, role):
        '''
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, float):
                if index.row() == 21:
                    return f"{value:.3f}mm"
                if index.row() > 13 and index.row() < 21:
                    return f"{value:.2f}%"
                elif index.row() > 21:
                    return f"{value:.4f}"
                else:
                    return f"{value:.6f}"
            return str(value)
        '''
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return value

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._headers[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
            

        

class MainWindow(QMainWindow):
    
    def show_RR_table_window(self, data):
        self.plot_setup_window = TableWindow(self, data)
        self.plot_setup_window.setWindowTitle("RR Data")
        self.plot_setup_window.show()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.settings = QSettings("PM Development", "Hydro Read Tag and R&R Tool")

        self.setWindowTitle("Get Tag Data/R&R")

        self.layout = QVBoxLayout()

        self.tag_input = QLineEdit()
        self.ip_input = QLineEdit()
        self.read_tag_button = QPushButton("Read Tag")
        self.RR_input = QLineEdit()
        self.RR_Browse_button = QPushButton('Browse')
        self.select_measurements_button = QPushButton('Select Measurements')
        self.RR_layout = QHBoxLayout()
        self.file_name_input = QLineEdit()
        self.RR_button = QPushButton("Run RR")
        
        self.CGK_input = QLineEdit()
        self.CGK_Browse_button = QPushButton('Browse')
        self.CGK_layout = QHBoxLayout()
        self.select_cgk_button = QPushButton('Select CGK')
        self.CGK_button = QPushButton("Run CGK")

        self.RR_layout.addWidget(self.RR_input)
        self.RR_layout.addWidget(self.RR_Browse_button)
        self.RR_layout.addWidget(self.select_measurements_button)

        

        self.CGK_layout.addWidget(self.CGK_input)
        self.CGK_layout.addWidget(self.CGK_Browse_button)
        self.CGK_layout.addWidget(self.select_cgk_button)

        # checkbox
        self.boxplots = QCheckBox("Display Box Plots")
        self.scatterplots = QCheckBox("Display Scatter Plots")
        self.type1_mode = QCheckBox("One Part, One Nest")
        self.show_part_data = QCheckBox("Show Part Data")

        self.ip_input.setPlaceholderText("Enter PLC IP")
        self.tag_input.setPlaceholderText("Enter Tag")
        self.file_name_input.setPlaceholderText("Name of Saved File")
        self.RR_input.setPlaceholderText("Enter R&R CSV File Name")
        self.CGK_input.setPlaceholderText("Enter CGK CSV File Name")
        

        # size ip input to be able to handle 40 characters
        self.ip_input.setFixedWidth(400)

        self.layout.addWidget(self.ip_input)
        self.layout.addWidget(self.tag_input)
        self.layout.addWidget(self.file_name_input)
        self.layout.addWidget(self.read_tag_button)
        self.layout.addLayout(self.RR_layout)
        self.layout.addWidget(self.boxplots)
        self.layout.addWidget(self.scatterplots)
        self.layout.addWidget(self.type1_mode)
        self.layout.addWidget(self.show_part_data)
        self.layout.addWidget(self.RR_button)
        self.layout.addLayout(self.CGK_layout)
        self.layout.addWidget(self.CGK_button)


        self.read_history()
        
        self.read_tag_button.clicked.connect(
            lambda: self.read_tag_clicked(self.tag_input.text(), self.ip_input.text()))
        
        self.RR_button.clicked.connect(
            lambda: self.RR_clicked(self.RR_input.text()))
        
        self.select_measurements_button.clicked.connect(
            lambda: self.open_measurement_selector(self.RR_input.text()))
        
        
        
        self.CGK_button.clicked.connect(
            lambda: self.CGK_clicked(self.CGK_input.text()))
        self.select_cgk_button.clicked.connect(
            lambda: self.open_cgk_selector(self.CGK_input.text()))
        
        self.RR_Browse_button.clicked.connect(
            lambda: self.RR_input.setText(QFileDialog.getOpenFileName(self, 'Open File', 'c:\\', 'CSV Files (*.csv)')[0]))
        
        
        
        self.CGK_Browse_button.clicked.connect(
            lambda: self.CGK_input.setText(QFileDialog.getOpenFileName(self, 'Open File', 'c:\\', 'CSV Files (*.csv)')[0]))
        
        self.setFixedSize(self.layout.sizeHint())
        # Set central widget
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        

    def read_tag_clicked(self, tag_input, ip_input):
        read_tag(tag_input, ip_input, self.file_name_input.text())
        self.save_history()

    def RR_clicked(self, csv_name):
        try:
            data = run_RR(csv_name, boxplots=self.boxplots.isChecked(), scatterplots=self.scatterplots.isChecked(), type1=self.type1_mode.isChecked(), show_part_data=self.show_part_data.isChecked(), selected_measurements=getattr(self, 'selected_measurements', None))
            # transpose data
            data = data.T
            self.show_RR_table_window(data)
            self.save_history()
        except ValueError as e:
            QMessageBox.warning(self, 'Selection Required', str(e))
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def CGK_clicked(self, csv_name):
        self.save_history()
        try:
            data = run_CGK(csv_name, selected_cgk=getattr(self, 'selected_cgk', None))
            print(data)
            data = data.T
            self.show_RR_table_window(data)
            self.save_history()
        except ValueError as e:
            QMessageBox.warning(self, 'Selection Required', str(e))
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def read_history(self):
        self.ip_input.setText(self.settings.value('ip', ''))
        self.tag_input.setText(self.settings.value('tag', ''))
        self.RR_input.setText(self.settings.value('RR', ''))
        # restore selected measurements if present
        try:
            stored = self.settings.value('selected_measurements', '')
            if stored:
                # stored as name:tolerance|name:tolerance|...
                items = [s for s in stored.split('|') if s]
                self.selected_measurements = [(i.split(':')[0], float(i.split(':')[1])) for i in items]
            else:
                self.selected_measurements = None
        except Exception:
            self.selected_measurements = None
        
        self.CGK_input.setText(self.settings.value('CGK', ''))
        # restore dynamic CGK selections
        try:
            stored_cgk = self.settings.value('selected_cgk', '')
            if stored_cgk:
                items = [s for s in stored_cgk.split('|') if s]
                self.selected_cgk = [(i.split(':')[0], float(i.split(':')[1]), float(i.split(':')[2])) for i in items]
            else:
                self.selected_cgk = None
        except Exception:
            self.selected_cgk = None

    def save_history(self):
        self.settings.setValue('ip', self.ip_input.text())
        self.settings.setValue('tag', self.tag_input.text())
        self.settings.setValue('RR', self.RR_input.text())
        # persist selected measurements
        if hasattr(self, 'selected_measurements') and self.selected_measurements:
            serialized = '|'.join([f"{name}:{tol}" for name, tol in self.selected_measurements])
            self.settings.setValue('selected_measurements', serialized)
        else:
            self.settings.setValue('selected_measurements', '')

    def open_measurement_selector(self, csv_path):
        if not csv_path:
            return
        try:
            df = pd.read_csv(csv_path, nrows=1)
        except Exception:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle('Select Measurements and Tolerances')
        layout = QVBoxLayout(dialog)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['Measurement (CSV Column)', 'Tolerance'])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        # Preload existing selections
        preselected = getattr(self, 'selected_measurements', []) or []
        pre_map = {name: tol for name, tol in preselected}

        # Populate rows with all numeric-eligible columns except index-like
        columns = [c for c in df.columns if c not in ['Index']]
        table.setRowCount(len(columns))
        for row, col_name in enumerate(columns):
            name_item = QTableWidgetItem(col_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)

            tol_value = pre_map.get(col_name, '')
            tol_item = QTableWidgetItem(str(tol_value) if tol_value != '' else '')
            table.setItem(row, 1, tol_item)

        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def accept():
            selections = []
            for r in range(table.rowCount()):
                name = table.item(r, 0).text() if table.item(r, 0) else ''
                tol_text = table.item(r, 1).text() if table.item(r, 1) else ''
                if name and tol_text:
                    try:
                        tol = float(tol_text)
                        selections.append((name, tol))
                    except ValueError:
                        pass
            self.selected_measurements = selections if selections else None
            dialog.accept()

        def reject():
            dialog.reject()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(reject)

        dialog.exec()
        
        self.settings.setValue('CGK', self.CGK_input.text())

    def open_cgk_selector(self, csv_path):
        if not csv_path:
            return
        try:
            df = pd.read_csv(csv_path, nrows=1)
        except Exception:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle('Select CGK Columns, Tolerances, Known Values')
        layout = QVBoxLayout(dialog)

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['Measurement (CSV Column)', 'Tolerance', 'Known Value'])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

        # Preload existing selections
        preselected = getattr(self, 'selected_cgk', []) or []
        pre_map = {name: (tol, known) for name, tol, known in preselected}

        columns = [c for c in df.columns if c not in ['Index']]
        table.setRowCount(len(columns))
        for row, col_name in enumerate(columns):
            name_item = QTableWidgetItem(col_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)

            tol_value = pre_map.get(col_name, ('' , ''))[0]
            known_value = pre_map.get(col_name, ('' , ''))[1]
            tol_item = QTableWidgetItem(str(tol_value) if tol_value != '' else '')
            known_item = QTableWidgetItem(str(known_value) if known_value != '' else '')
            table.setItem(row, 1, tol_item)
            table.setItem(row, 2, known_item)

        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def accept():
            selections = []
            for r in range(table.rowCount()):
                name = table.item(r, 0).text() if table.item(r, 0) else ''
                tol_text = table.item(r, 1).text() if table.item(r, 1) else ''
                known_text = table.item(r, 2).text() if table.item(r, 2) else ''
                if name and tol_text and known_text:
                    try:
                        tol = float(tol_text)
                        known = float(known_text)
                        selections.append((name, tol, known))
                    except ValueError:
                        pass
            self.selected_cgk = selections if selections else None
            dialog.accept()

        def reject():
            dialog.reject()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(reject)

        dialog.exec()

        # persist selection now
        if hasattr(self, 'selected_cgk') and self.selected_cgk:
            serialized = '|'.join([f"{name}:{tol}:{known}" for name, tol, known in self.selected_cgk])
            self.settings.setValue('selected_cgk', serialized)
        else:
            self.settings.setValue('selected_cgk', '')
        # persist selected CGK
        if hasattr(self, 'selected_cgk') and self.selected_cgk:
            serialized = '|'.join([f"{name}:{tol}:{known}" for name, tol, known in self.selected_cgk])
            self.settings.setValue('selected_cgk', serialized)
        else:
            self.settings.setValue('selected_cgk', '')
        self.settings.setValue('CGK', self.CGK_input.text())

app = QApplication(sys.argv)
app.processEvents()
app.setWindowIcon(QtGui.QIcon('icon.ico'))
qdarktheme.setup_theme()
window = MainWindow()
window.resize(1000, 800)
window.show()

app.exec()
