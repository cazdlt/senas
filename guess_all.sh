for i in {1,2,3,5,6,7,12,13,15,17,18,19,27,28,29,38,40,41,43,48,49,50,51,52}
do
echo -n "Seña  $i: "
python classify.py $i
done
