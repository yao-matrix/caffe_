#!/bin/sh

# From Intel Cooperation

# check cpu name
CPU_NAME=`cat /proc/cpuinfo |grep "model name"|awk -F ':' '{print $2}'|uniq`
echo "-----------------CPU NAME -------------------"
echo "$CPU_NAME"

# check DDR
RAM_SIZE=`free -h`
echo "----------------MEMORY SIZE -----------------"
echo "$RAM_SIZE"

echo "-------------MEMORY CLOCK SPEED --------------"
RAM_CLOCK=`dmidecode -t memory|grep -ie HZ`
echo "$RAM_CLOCK"

# check HT
HT=`lscpu|grep "per core"|awk -F ':' '{print $2}'|sed 's/^ *\| *$//g'`
echo "-------------------  HT ---------------------"
if [[ ${HT} == 1 ]]; then
	echo "HT:OFF"
elif [[ ${HT} > 1 ]]; then
	echo "HT:ON"
else
	echo "HT:UNKHOWN"
fi

# check TURBO
TURBO=`cat /sys/devices/system/cpu/intel_pstate/no_turbo`
echo "-------------------TURBO---------------------"
if [[ ${TURBO} == 1 ]]; then
	echo "TURBO:OFF"
else
	echo "TURBO:On"
fi

# check threads env
echo "-------------------THREADS ENV---------------------"
MKL_NUM_THREADS=`env |grep MKL_NUM_THREADS`
if [ "${MKL_NUM_THREADS}" = "" ]; then
	echo "MKL_NUM_THREADS:UNSET"
else
	echo "$MKL_NUM_THREADS"
fi

OMP_NUM_THREADS=`env |grep OMP_NUM_THREADS`
if [ "${OMP_NUM_THREADS}" = "" ]; then
	echo "OMP_NUM_THREADS:UNSET"
else
	echo "$OMP_NUM_THREADS"
fi

KMP_AFFINITY=`env |grep KMP_AFFINITY`
if [ "${KMP_AFFINITY}" = "" ]; then
	echo "KMP_AFFINITY:unset"
else
	echo "$KMP_AFFINITY"
fi

# check CPU power governor
echo "-------------------POWER GOVERNOR---------------------"
POWER_GOVERNOR=`cpupower frequency-info` | grep 
echo $POWER_GOVERNOR
if [ "${POWER_GOVERNOR}" = "" ]; then
	echo "KMP_AFFINITY:unset"
else
	echo "$KMP_AFFINITY"
fi

# check NUMA and MCDRAM configuration
# You should yum install hwloc
echo "-------------------NUMA and MCDRAM mode---------------------"
POWER_GOVERNOR=`hwloc` | grep 
echo $POWER_GOVERNOR
if [ "${POWER_GOVERNOR}" = "" ]; then
	echo "KMP_AFFINITY:unset"
else
	echo "$KMP_AFFINITY"
fi


