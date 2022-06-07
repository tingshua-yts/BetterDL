horovodrun  --verbose --network-interface "eth0" --log-level DEBUG \
	    -p 12345\
	    -np 4\
	    -H 192.168.9.106:2,192.168.9.104:2\
	    /usr/bin/python ./train_dist.py
