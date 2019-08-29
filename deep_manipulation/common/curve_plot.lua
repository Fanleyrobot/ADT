--[[ File
@description:
    The functions included in this file are for ploting curves.
@version: V0.10
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   23/06/2015  developed the first version
    V0.10   26/07/2018  minor updates for code re-organization
]]


require "torch"
require "gnuplot"


-- Visualize a tensor into a line and save as a png file
function plotTensorToFile(tensor, tensor_name, x_label, y_label, file_name)
    gnuplot.pngfigure(file_name)
    gnuplot.plot(
        {tensor_name, tensor, '-'}
        )
    gnuplot.xlabel(x_label)
    gnuplot.ylabel(y_label)
end

-- Visualize a tensor into a line and display in a window numberred as figure_number
function plotTensorToDisplay(tensor, tensor_name, x_label, y_label, figure_number)
    gnuplot.figure(figure_number)
    gnuplot.plot(
        {tensor_name, tensor, '-'}
        )
    gnuplot.xlabel(x_label)
    gnuplot.ylabel(y_label)
end
