#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

# Logging in
echo 'To download the Human3.6M dataset, you first need to register on the official website at http://vision.imar.ro/human3.6m'
echo "If that's done, enter your details below:"
printf 'Email registered on the Human3.6M website: '
read -r email
printf 'Password: '
read -rs password
encoded_email=$(urlencode "$email")

login_url="https://vision.imar.ro/human3.6m/checklogin.php"
download_url="http://vision.imar.ro/human3.6m/filebrowser.php"
cookie_path=$(mktemp)

_term() {
  # Make sure to clean up the cookie file
  rm "$cookie_path"
}
trap _term SIGTERM SIGINT

curl "$login_url" --insecure --verbose --data "username=$encoded_email&password=$password" --cookie-jar "$cookie_path" --cookie "$cookie_path"

get_file() {
  curl --remote-name --remote-header-name --verbose --cookie-jar "$cookie_path" --cookie "$cookie_path" "$1"
}

get_subject_data() {
  get_file "$download_url?download=1&filepath=Videos&filename=SubjectSpecific_$1.tgz&downloadname=$2"
  get_file "$download_url?download=1&filepath=Segments/mat_gt_bb&filename=SubjectSpecific_$1.tgz&downloadname=$2"
  get_file "$download_url?download=1&filepath=Poses/D3_Positions&filename=SubjectSpecific_$1.tgz&downloadname=$2"
}

mkdir -p "$DATA_ROOT/h36m"
cd "$DATA_ROOT/h36m" || exit

get_subject_data 1 S1
get_subject_data 6 S5
get_subject_data 7 S6
get_subject_data 2 S7
get_subject_data 3 S8
get_subject_data 4 S9
get_subject_data 5 S11
get_file http://vision.imar.ro/human3.6m/code-v1.2.zip

for i in 1 5 6 7 8 9 11; do
  tar -xvf "Videos_S$i.tgz"
  rm "Videos_S$i.tgz"
  tar -xvf "Segments_mat_gt_bb_S$i.tgz"
  rm "Segments_mat_gt_bb_S$i.tgz"
  tar -xvf "Poses_D3_Positions_S$i.tgz"
  rm "Poses_D3_Positions_S$i.tgz"
done

unzip code-v1.2.zip
rm code-v1.2.zip
