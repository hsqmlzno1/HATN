#!/bin/sh

domains=('books' 'dvd' 'electronics' 'kitchen' 'video')
for src_domain in ${domains[@]};
do
	for tar_domain in  ${domains[@]};
	do
		if [ $src_domain != $tar_domain ];
		then
			python extract_pivots.py --train --test -s $src_domain -t $tar_domain -v | tee -a ./work/logs/PNet.txt
			python train_hatn.py     --train --test -s $src_domain -t $tar_domain -v | tee -a ./work/logs/HATN.txt
		fi
	done
done


