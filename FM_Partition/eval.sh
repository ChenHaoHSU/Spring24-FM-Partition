MY_DIR='output/ch48458'
REF_DIR='output/reference'


for i in {0..10}
do
  CASE='example_$i.txt'
  diff ${MY_DIR}/${CASE} ${REF_DIR}/${CASE}
  ret=$?
  if [ ret 
done

