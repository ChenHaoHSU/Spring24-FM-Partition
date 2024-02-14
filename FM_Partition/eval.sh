MY_DIR="output/ch48458"
REF_DIR="output/reference"

for i in {0..20}
do
  CASE="example_$i.txt"
  diff ${MY_DIR}/${CASE} ${REF_DIR}/${CASE}
  ret=$?
  if [ $ret -eq 0 ]; then
    echo "[Info] ${CASE} same! :)"
  else
    echo "[Error] ${CASE} different.. :("
  fi
done

