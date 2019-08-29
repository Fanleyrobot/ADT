--[[ File
@description:
    The functions included in this file are for ploting customized curves.
    The codes here are used for ICRA 2017 and ACRA 2017,
    not used in the current framework.
@version: V0.10
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   27/08/2015  developed the first version
    V0.10   26/07/2018  minor changes for code re-organization
]]


require 'torch'
require 'gnuplot'


-- construct a class
local dpr = torch.class('data_processing')


--[[ Function
@description: initialize an object for curve plotting
@input:
    args: settings for a curve plotting object
@output: nil
@notes:
]]
function dpr:__init(args)
    -- self.font =
    self.color = {}
    -- self.dash_type = {1, 3, 4, }
    self.line_style = {'line ls 1', 'line ls 2', 'line ls 3', 'line ls 4', 'line ls 5'}
    -- self.line_style = {'-', '-', '-', '-', '-'}
    -- self.line_style = {'line lt 1', 'line lt 2', 'line lt 3', 'line  lt 4', 'line lt 5'}

end

function dpr:plotQcurve(data1, x_label, y_label, file_name)
    gnuplot.pdffigure(file_name)

    -- set image format
    gnuplot.raw("set terminal pdf font 'Times-New-Roman, 16'")
    gnuplot.raw("set terminal pdf size 4.5,3")
    gnuplot.raw("set terminal pdf dashed")
    gnuplot.raw("set for [i=1:5] linetype i dt i")
    gnuplot.raw("set style line 1 lt 1 lc rgbcolor blue_050  lw 5")
    gnuplot.raw("set style line 2 lt 2 lc rgbcolor green_050  lw 5")
    gnuplot.raw("set style line 3 lt 3 lc rgbcolor red_050  lw 5")
    gnuplot.raw("set style line 4 lt 4 lc rgbcolor brown_050  lw 5")
    gnuplot.raw("set style line 5 lt 5 lc rgbcolor blue_025  lw 5")

    gnuplot.plot(data1, '-')
    gnuplot.xlabel(x_label)
    gnuplot.ylabel(y_label)
    -- gnuplot.movelegend('right', 'bottom')
end

-- function dpr:plotTensorToFile(tensor, curve_name, x_label, y_label, file_name)
--     gnuplot.pdffigure(file_name)
--
--     -- set image format
--     gnuplot.raw("set terminal pdf font 'Times-New-Roman, 16'")
--     gnuplot.raw("set terminal pdf size 4.5,3")
--     gnuplot.raw("set terminal pdf dashed")
--     gnuplot.raw("set for [i=1:5] linetype i dt i")
--     gnuplot.raw("set style line 1 lt 1 lc rgbcolor blue_050  lw 5")
--     gnuplot.raw("set style line 2 lt 2 lc rgbcolor green_050  lw 5")
--     gnuplot.raw("set style line 3 lt 3 lc rgbcolor red_050  lw 5")
--     gnuplot.raw("set style line 4 lt 4 lc rgbcolor brown_050  lw 5")
--     gnuplot.raw("set style line 5 lt 5 lc rgbcolor blue_025  lw 5")
--
--     gnuplot.plot(
--         {curve_name[1], tensor[{1,{}}], self.line_style[1]},
--         {curve_name[2], tensor[{2,{}}], self.line_style[2]},
--         {curve_name[3], tensor[{3,{}}], self.line_style[3]},
--         {curve_name[4], tensor[{4,{}}], self.line_style[4]},
--         {curve_name[5], tensor[{5,{}}], self.line_style[5]})
--     gnuplot.xlabel(x_label)
--     gnuplot.ylabel(y_label)
--     gnuplot.movelegend('right', 'bottom')
-- end

function dpr:boxplotStepDataToFile(data_name, x_label, y_label, file_name)
    gnuplot.pdffigure(file_name)
    -- gnuplot.figure(9)
    -- gnuplot.raw('set output "boxplot.pdf"')
    gnuplot.raw("set terminal pdf font 'Times-New-Roman, 16'")
    gnuplot.raw("set terminal pdf size 3.5,3")
    -- gnuplot.raw("set terminal pdf dashed")
    gnuplot.raw("set style fill solid 0.25 border -1")
    gnuplot.raw("set style boxplot outliers pointtype 1")
    gnuplot.raw("set style data boxplot")
    -- gnuplot.raw("set boxwidth 0.7 absolute")
    gnuplot.raw("set datafile separator '\t'")
    -- gnuplot.raw("set xtics ('' 0, 'A' 1, 'B' 2, 'C' 3, 'D' 4, 'E' 5, '' 6) scale 0")
    -- gnuplot.raw("set xrange[0:6]")
    gnuplot.raw("set xtics ('' 0, '1' 1, '2' 2, '3' 3, '' 4) scale 0")
    gnuplot.raw("set xrange[0:4]")

    gnuplot.raw(string.format("plot for [i=1:3] '%s' using (i):i notitle", data_name))

    gnuplot.xlabel(x_label)
    gnuplot.ylabel(y_label)
end

function dpr:boxplotStepTensorToFile(tensor, x_label, y_label, file_name)
    self:saveTensorToTxt(tensor, 'temp_data1.txt')
    self:boxplotStepDataToFile('temp_data1.txt', x_label, y_label, file_name)
end

function dpr:boxplotTestDataToFile(data_name, x_label, y_label, file_name)
    gnuplot.pdffigure(file_name)
    -- gnuplot.figure(9)
    -- gnuplot.raw('set output "boxplot.pdf"')
    gnuplot.raw("set terminal pdf font 'Times-New-Roman, 16'")
    gnuplot.raw("set terminal pdf size 3.5,3")
    -- gnuplot.raw("set terminal pdf dashed")
    gnuplot.raw("set style fill solid 0.25 border -1")
    gnuplot.raw("set style boxplot outliers pointtype 1")
    gnuplot.raw("set style data boxplot")

    gnuplot.raw("set style line 3 lt 3 lc rgbcolor red_25")
    gnuplot.raw("set style line 2 lt 4 lc rgbcolor green_25")
    gnuplot.raw("set style line 1 lt 5 lc rgbcolor blue_25")


    -- gnuplot.raw("set boxwidth 0.7 absolute")
    gnuplot.raw("set datafile separator '\t'")
    -- gnuplot.raw("set xtics ('' 0, 'A' 1, 'B' 2, 'C' 3, 'D' 4, 'E' 5, '' 6) scale 0")
    -- gnuplot.raw("set xrange[0:6]")
    gnuplot.raw("set xtics ('' 0, '1' 1, '2' 2, '3' 3, '' 4) scale 0")
    gnuplot.raw("set xrange[0:4]")

    gnuplot.raw(string.format("plot for [i=1:3] '%s' using (i):i notitle", data_name))

    gnuplot.xlabel(x_label)
    gnuplot.ylabel(y_label)
end

function dpr:boxplotTestTensorToFile(tensor, x_label, y_label, file_name)
    self:saveTensorToTxt(tensor, 'temp_data2.txt')
    self:boxplotTestDataToFile('temp_data2.txt', x_label, y_label, file_name)
end


function dpr:boxplotDataToFile(data_name, x_label, y_label, file_name)
    gnuplot.pdffigure(file_name)
    -- gnuplot.figure(9)
    -- gnuplot.raw('set output "boxplot.pdf"')
    gnuplot.raw("set terminal pdf font 'Times-New-Roman, 16'")
    gnuplot.raw("set terminal pdf size 3.5,3")
    -- gnuplot.raw("set terminal pdf dashed")
    gnuplot.raw("set style fill solid 0.25 border -1")
    gnuplot.raw("set style boxplot outliers pointtype 7")
    gnuplot.raw("set style data boxplot")
    -- gnuplot.raw("set boxwidth 0.7 absolute")
    gnuplot.raw("set datafile separator '\t'")
    gnuplot.raw("set xtics ('' 0, 'A' 1, 'B' 2, 'C' 3, 'D' 4, 'E' 5, '' 6) scale 0")
    gnuplot.raw("set xrange[0:6]")

    gnuplot.raw(string.format("plot for [i=1:5] '%s' using (i):i notitle", data_name))

    gnuplot.xlabel(x_label)
    gnuplot.ylabel(y_label)
end


function dpr:boxplotTensorToFile(tensor, x_label, y_label, file_name)
    self:saveTensorToTxt(tensor, 'temp_data.txt')
    self:boxplotDataToFile('temp_data.txt', x_label, y_label, file_name)
end


-- function dpr:plotTensorToDisplay(tensor, tensor_name, x_label, y_label, figure_number)
--     gnuplot.figure(figure_number)
--     gnuplot.plot(
--         {tensor_name, tensor, '-'}
--         -- {'td_history', ind, td_history[ind], '-'}
--         -- {'qmax_history', ind, qmax_history[ind], '-'}
--         )
--     gnuplot.xlabel(x_label)
--     gnuplot.ylabel(y_label)
-- end

function dpr:saveTensorToTxt(tensor, file_name)
    dim_ = #tensor
    -- os.remove(file_name)
    file_ = io.open(file_name, "w") -- overwrite an existing file or create a new file

    file_:write("# Agent_A\tAgent_B\tAgent_C\tAgent_D\tAgent_E")
    for i=1,dim_[2] do
        file_:write("\n")
        for j=1,dim_[1] do
            file_:write(tensor[j][i].."\t")
        end
    end

    io.close(file_)
end

function dpr:saveMeanToTxt(mean, variance, success_rate, file_name)
    -- os.remove(file_name)
    file_ = io.open(file_name, "w") -- overwrite an existing file or create a new file

    file_:write("R_Mean\tR_Variance\tSuccess_Rate\n")
    file_:write(mean.."\t"..variance.."\t"..success_rate.."\n")

    io.close(file_)
end

function dpr:saveStatisticToTxt(mean, variance, data_type, file_name)
    -- os.remove(file_name)
    file_ = io.open(file_name, "w") -- overwrite an existing file or create a new file

    file_:write("Mean\tVariance\tData Type\n")
    file_:write(mean.."\t"..variance.."\t"..data_type.."\n")

    io.close(file_)
end
