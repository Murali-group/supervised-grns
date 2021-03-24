#!/bin/bash
declare -a datasets=("/home/malabika/GRN/hsc_combined1k.csv")
declare -a networks=("/home/malabika/GRN/STRINGc700.csv")
declare -a encoder=("GCN" "DGCN")
declare -a decoder=("IP" "TF" "RS")

for enc in ${encoder[*]}; do
    for dec in ${decoder[*]}; do
        for eFile in ${datasets[*]}; do
            for nFile in ${networks[*]}; do
                for l in {2..2}; do
                    for kT in {1..1}; do
                        for j in {0..4}; do
                            for k in {1..2}; do
                                i=$((2*j+k))
                                echo "screen -S testCV$i -d -m /bin/sh -c \"/home/malabika/anaconda3/condabin/conda activate grnenv; python runGCN-CV.py -e=$eFile -n=$nFile --epochs=20  --kTest=$kT --kTrain=$kT --type=E --encoder=$enc --decoder=$dec --rand=2019 --id=$i -l=$l -i >> ../outputs/GCN-CV-BM.out\""
                                screen -S testCV$i -d -m /bin/sh -c "home/malabika/anaconda3/condabin/conda activate grnenv; python runGCN-CV.py -e=$eFile -n=$nFile --epochs=20 --kTest=$kT --kTrain=$kT --type=E --encoder=$enc --decoder=$dec --rand=2019 --id=$i -l=$l -i >> ../outputs/GCN-CV-BM.out"
                                pids[${i}]=$!
                            done
                            while screen -list | grep -q testCV;
                            do
                                sleep 1
                            done
                        done
                    done
                done
            done
        done
    done
done
