class ArticlesController < ApplicationController

	def index
		respond_to do |format|
			format.html
			format.json
		end
	end

	def new
		@article = Article.new
	end

	def create
		@article = Article.new(article_params)
		uri = URI.parse("http://0.0.0.0:8000")
		http = Net::HTTP.new(uri.host, uri.port)
		request = Net::HTTP::Post.new("/predict")
		response = http.request(request)

		@aux = Article.new
		@aux.title = response.body	
		@aux.text = response.body

 		respond_to do |format|
 			if @article.save
 				format.js {render 'show'}	
 			end
 		end
  		
	end

	def show
	end

	private
		def article_params
    		params.require(:article).permit(:title, :text)
  		end
end
