class ImagesController < ApplicationController
  BOUNDARY = "AaB03x"
  def index
  end

  def new
    @image = Image.new
  end

  def create
    @image = Image.create(image_params)

    respond_to do |format|
      if @image.save
        path = "#{Rails.root}/public/image_trans/" + @image.id.to_s + "/" + @image.text_image_file_name 
        #response = RestClient.post 'http://0.0.0.0:9000/get_sentence', :imgProcessing => File.new(path, 'rb')

        @aux = Ann.new


        format.js {render 'show'} 
      end
    end
  end

  def random
    @image = Image.order("RANDOM()").first
    if @image
      redirect_to edit_image_path(@image.id)
    else
      redirect_to new_image_path
    end
  end

  def edit
    @image = Image.find(params[:id])
    @path = "/image_trans/" + @image.id.to_s + "/" + @image.text_image_file_name
  end

  def update
  end

  private
    def image_params
      params.require(:image).permit(:text_image)
    end

end
