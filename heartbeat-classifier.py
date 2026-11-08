import wfdb
from IPython.display import display

record = wfdb.rdrecord('../mitbih/100', sampto=15000)
ann = wfdb.rdann('../mitbih/100', 'atr', sampto=15000)

#wfdb.plot_wfdb(record=record, annotation=ann, title="Record 100 from MIT-BIH", time_units='seconds')
display(record.__dict__)
display(ann.__dict__)