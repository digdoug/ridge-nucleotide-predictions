#!/bin/bash
X=180
TOTAL=310
while [ $X -lt $TOTAL ]
do
	PBS=bash_scripts/bash_$X.txt 
	cp runMultiClassTemp.sh $PBS
	X1=$((X+10))
	X2=$((X+1))
	X3=$((X+11))
	echo -n "python ml_pipeline_wMLP.py hs_ref_GRCh38_chr22.fa good_results/DTClass/class_${X}_$X1.txt $X2 $X3" >> $PBS
							        
	sbatch -C avx $PBS
	X=$((X+10))					
done
