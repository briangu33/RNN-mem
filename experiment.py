from simple_rnn import Simple_RNN
from data_utils import acc_print
from data_utils import char_printer
from data_utils import acc_count
from data_utils import char_counter

def char_print_experiment():
	char_printer(100000)

	rnn = Simple_RNN(2, 12)
	rnn.train("data/char_printer_input9.txt", 100001)

	sample, hs = rnn.sample(n=10000)
	print acc_print(sample)

def char_count_experiment():
	char_counter(100000)

	rnn = Simple_RNN(10, 12)
	rnn.train("data/char_counter_input9.txt", 100001)

	sample, hs = rnn.sample(n=10000)
	print "Accuracy:", acc_count(sample)

char_print_experiment()