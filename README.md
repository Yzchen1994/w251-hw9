# W251 HW 9

## Setup a pair of 2xV-100 VMs in IBM cloud
1. In IBM Cloud base machine, Run
```
ibmcloud sl vs create --datacenter=lon04 --hostname=v100a --domain=hw9.UC-Berkeley.cloud --image=2263543 --billing=hourly  --network 1000 --key=<YOUR-KEY-ID> --flavor AC2_16X120X100 --san

ibmcloud sl vs create --datacenter=lon04 --hostname=v100b --domain=hw9.UC-Berkeley.cloud --image=2263543 --billing=hourly  --network 1000 --key=<YOUR-KEY-ID> --flavor AC2_16X120X100 --san
```
2. Add 2TB disk space for each VM. 
3. Mount the disk inside each VM
```
root@v100a:~# fdisk -l
Disk /dev/xvdh: 64 MiB, 67125248 bytes, 131104 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0x00000000


Disk /dev/xvdb: 2 GiB, 2147483648 bytes, 4194304 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0x00025cdb

Device     Boot Start     End Sectors Size Id Type
/dev/xvdb1         63 4192964 4192902   2G 82 Linux swap / Solaris


Disk /dev/xvda: 100 GiB, 107374182400 bytes, 209715200 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0x5dfd6b86

Device     Boot  Start       End   Sectors  Size Id Type
/dev/xvda1 *      2048    526335    524288  256M 83 Linux
/dev/xvda2      526336 209715166 209188831 99.8G 83 Linux


Disk /dev/xvdc: 2 TiB, 2147483648000 bytes, 4194304000 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
root@v100a:~# mkdir -m 777 /data
root@v100a:~# mkfs.ext4 /dev/xvdc
mke2fs 1.42.13 (17-May-2015)
Creating filesystem with 524288000 4k blocks and 131072000 inodes
Filesystem UUID: 3d596872-24dc-40d3-82f0-4e5e43db8f63
Superblock backups stored on blocks:
	32768, 98304, 163840, 229376, 294912, 819200, 884736, 1605632, 2654208,
	4096000, 7962624, 11239424, 20480000, 23887872, 71663616, 78675968,
	102400000, 214990848, 512000000

Allocating group tables: done
Writing inode tables: done
Creating journal (32768 blocks): done
Writing superblocks and filesystem accounting information: done

root@v100a:~# nano /etc/fstab
root@v100a:~# mount /data
root@v100a:~# service docker stop
root@v100a:~# cd /var/lib
root@v100a:/var/lib# cp -r docker /data
root@v100a:/var/lib# rm -fr docker
root@v100a:/var/lib# ln -s /data/docker ./docker
root@v100a:/var/lib# service docker start

```

4. Clone hw repo into the VM
```
root@v100a:~# git clone git@github.com:MIDS-scaling-up/v2.git
```

## Create cloud containers for openseq2seq and distributed training
1. Setup docker image in both VMs.
```
root@v100a:~# cd v2/week09/hw/docker/
root@v100a:~/v2/week09/hw/docker# docker image build -t openseq2seq .
```
2. Setup docker container in both VMs. SSH into the container. 
```
root@v100a:~/v2/week09/hw/docker# docker run --runtime=nvidia -d --name openseq2seq --net=host -e SSH_PORT=4444 -v /data:/data -p 6006:6006 openseq2seq
root@v100a:~/v2/week09/hw/docker# docker exec -ti openseq2seq bash
```
3. Inside container, test mpi: 
```
root@v100a:~# mpirun -n 2 -H 10.222.16.2,10.222.16.6 --allow-run-as-root hostname
[v100a:00079] WARNING: local probe returned unhandled shell:unknown assuming bash
Warning: Permanently added '[10.222.16.6]:4444' (ECDSA) to the list of known hosts.
rm: cannot remove '/lib': Is a directory
v100a
v100b
```
4. Pull data to be used in neural machine tranlsation training. This will take about 1 hours. 
```
root@v100a:~# cd /opt/OpenSeq2Seq
root@v100a:/opt/OpenSeq2Seq# scripts/get_en_de.sh /data/wmt16_de_en
```

5. Copy configuration file to /data directory: 
```
cp /opt/OpenSeq2Seq/example_configs/text2text/en-de/transformer-base.py /data
```

6. Edit the transformer-base.py as indicated inside this repo. 
7. Start training on the ** first VM only**. 
```
root@v100a:/data# cd /opt/OpenSeq2Seq/
root@v100a:/opt/OpenSeq2Seq# nohup mpirun --allow-run-as-root -n 4 -H 10.222.16.2:2, 10.222.16.6:2 -bind-to none -map-by slot --mca btl_tcp_if_include eth0 -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH python run.py --config_file=/data/transformer-base.py --use_horovod=True --mode=train_eval &
```
8. SSH into VM1, ssh into the docker container. Monitor the training progress
```
cd /opt/OpenSeq2Seq
tail -f nohup.out
```
9. Start tensorboard on the same machine where you started training. Then monitor the progress at http://public_ip_of_your_vm1:6006
```
nohup tensorboard --logdir=/data/en-de-transformer
```
10. Setup nmon in both VMs to monitor the network performance
```
apt install nmon
```

## Q&A
### How long does it take to complete the training run? (hint: this session is on distributed training, so it will take a while)
It takes 17.5 hours. But I cut it short to only run 42900 steps. 

### Do you think your model is fully trained? How can you tell?
No. The score becomes 0 for some reason. 

### Were you overfitting?
Possibly. The Eval loss becomes NaN after 4000 steps, so it might be reaching to a local minimum.  ![Alt text](https://raw.githubusercontent.com/Yzchen1994/w251-hw9/master/Screen%20Shot%202020-06-29%20at%205.36.29%20PM.png)

### Were your GPUs fully utilized?
Yes. As shown in nvidia-smi GPU usage, it's always 100% GPU-Util. 
![Alt text](https://raw.githubusercontent.com/Yzchen1994/w251-hw9/master/v100a-gpu-1.png "V100a GPU Utilization")
![Alt text](https://raw.githubusercontent.com/Yzchen1994/w251-hw9/master/v100b-gpu-1.png "V100b GPU Utilization")

### Did you monitor network traffic (hint: apt install nmon ) ? Was network the bottleneck?
Network doesn't seem to have any bottleneck as shown in the nmon monitoring during the training. 
![Alt text](https://raw.githubusercontent.com/Yzchen1994/w251-hw9/master/v100a-network-1.png "V100a Network Utilization 1")
![Alt text](https://raw.githubusercontent.com/Yzchen1994/w251-hw9/master/v100a-network-2.png "V100a Network Utilization 2")
![Alt text](https://raw.githubusercontent.com/Yzchen1994/w251-hw9/master/v100b-network-1.png "V100a Network Utilization 1")
![Alt text](https://raw.githubusercontent.com/Yzchen1994/w251-hw9/master/v100b-network-2.png "V100a Network Utilization 2")

### Take a look at the plot of the learning rate and then check the config file. Can you explan this setting?
Learning rate is going up and then down. That is expected. 

### How big was your training set (mb)? How many training lines did it contain?
~959MB. 4.5 million lines. 

### What are the files that a TF checkpoint is comprised of?
![Alt text](https://raw.githubusercontent.com/Yzchen1994/w251-hw9/master/best_model_folder.png)

### How big is your resulting model checkpoint (mb)?
813M

### Remember the definition of a "step". How long did an average step take?


### How does that correlate with the observed network utilization between nodes?
