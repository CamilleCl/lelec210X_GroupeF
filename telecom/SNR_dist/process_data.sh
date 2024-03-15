for file in *.txt; do
    python3 read_file_for_CFO_SNR.py $file
done

rm -f *.txt