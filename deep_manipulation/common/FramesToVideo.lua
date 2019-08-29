--[[ File
@description:
	The functions included in this file are for generating videos
	through merging separate images in sequence.
@version: V0.02
@author: Fangyi Zhang	email:gzzhangfangyi@gmail.com
@acknowledgement:
	Adam Tow
	ARC Centre of Excellence for Robotic Vision (ACRV)
	Queensland Univsersity of Technology (QUT)
@history:
	V0.00	15/07/2015	developed the first version
	V0.01	07/08/2015	updated to support setting a start frame
	V0.02	07/08/2015	updated to set the video size to the input image size
]]


require "torch"


--[[ Function
@description: generate a video file (vfile) by merging images (ifile).
@input:
	ifile: the directory path and name format of image files
			i.e., /home/Manipulation/top_view%03d.png
	vfile: the directory path and file name of the video to be generated
			i.e., /home/Manipulation/top_view.mp4
	start: the start frame number, this parameter is optional
@output: nil
@notes: the sequence of images is determined by the image number, such as '%3d'.
]]
function GenerateVideo(ifile, vfile, start)
	local fps = 25 -- set the frame rate for the video 
	if not start then
		start = ''
	else
		start = ' -start_number ' .. start
	end

	-- construct a system command according to the inputs
	local ffmpeg_cmd = 'ffmpeg' 
	.. ' -framerate ' .. fps
	.. start
	--.. ' -pattern_type glob '
	.. ' -i ' .. ifile 
	.. ' -vf scale=iw:ih'
	.. ' -c:v libx264'
	.. ' -r ' .. fps 
	.. ' -pix_fmt yuv420p ' 
	.. vfile 
	--.. '.mp4' 
	.. ' 2> /dev/null'

	--ffmpeg_cmd = ffmpeg_cmd .. ' -sws_flags neighbor -vf scale=' .. zoom .. '*iw:' .. zoom .. '*ih -vcodec mjpeg -qscale 1 -an ' .. outpath .. '.avi'

	-- back up the old file
	if paths.filep(vfile) then
		print('WARNING: ' .. vfile .. ' exist and will be backed up as ' .. vfile .. '.old ...')
		os.execute('mv -rf ' .. vfile .. ' ' ..vfile .. '.old')
	end

	-- generate a new video
    --print('Creating video from frames ...')
    --print(ffmpeg_cmd)
    os.execute(ffmpeg_cmd)

	--print('Deleting frames ...')
	--os.execute('rm ' .. path .. '*.png')
end