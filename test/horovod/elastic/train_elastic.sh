horovodrun  -p 12345 --verbose -np 2 --log-level DEBUG --host-discovery-script ./discover_hosts.sh  python train_pytorch_elastic.py
# --network-interface "eth0,eth1" 这个设置存在问题，代码中需要set类型，但却按照string来下类接收的参数
