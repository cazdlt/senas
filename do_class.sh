for i in {1,15,7,18,38,40,3,6,13,43,49,51}
do
echo Seña $i
python train.py $i -i
done
