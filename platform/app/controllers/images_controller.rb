class ImagesController < ApplicationController
  BOUNDARY = "AaB03x"
  def index
  end

  def new
    @image = Image.new
  end

  def create
    @image = Image.create(image_params)
    
    @aux1 = Ann.new
    @aux1.x0 = 12 
    @aux1.y0 = 34
    @aux1.x1 = 100
    @aux1.y1 = 200

    

    

    respond_to do |format|
      if @image.save
        path = "#{Rails.root}/public/system/text_images/111/original_Captura_de_pantalla_de_2017-09-12_11-43-48.png" 
        response = RestClient.post 'http://0.0.0.0:9000/get_sentence', :imgProcessing => File.new(path, 'rb')

        @aux = Ann.new
        @aux.x0 = 12 
        @aux.y0 = 34
        @aux.x1 = 100
        @aux.y1 = 200
        @aux.word = JSON.parse(response.body)['word'][0]

        format.js {render 'show'} 
      end
    end
  end

  private
    def image_params
      params.require(:image).permit(:text_image)
    end

end
