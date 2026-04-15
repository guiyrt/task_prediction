REMOTE_HOST="tern@192.168.68.64"

set -e

cd ~/task_prediction
docker build --platform linux/amd64 -t task-pred:latest .
docker save -o task-pred.tar task-pred:latest
scp task-pred.tar {$REMOTE_HOST}:/home/tern/zhaw/
ssh {$REMOTE_HOST} "cd zhaw/; docker load -i task-pred.tar"

# cd ~/instance_pred
# docker build --platform linux/amd64 -t instance-pred:latest .
# docker save -o instance-pred.tar instance-pred:latest
# scp instance-pred.tar {$REMOTE_HOST}:/home/tern/zhaw/
# ssh {$REMOTE_HOST} "cd zhaw/; docker load -i instance-pred.tar"

# cd ~/screen_recording
# docker build --platform linux/amd64 -t screen-recorder:latest .
# docker save -o screen-recorder.tar screen-recorder:latest
# scp screen-recorder.tar $REMOTE_HOST:/home/tern/zhaw/
# ssh $REMOTE_HOST "cd zhaw/; docker load -i screen-recorder.tar"

# cd ~/gaze_capture
# docker build --platform linux/amd64 -t gaze-capture:latest .
# docker save -o gaze-capture.tar gaze-capture:latest
# scp gaze-capture.tar $REMOTE_HOST:/home/tern/zhaw/
# ssh $REMOTE_HOST "cd zhaw/; docker load -i gaze-capture.tar"