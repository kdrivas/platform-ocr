class Image < ApplicationRecord

  has_attached_file :text_image, :url  => "/image_trans/:id/:filename",
     :path => ":rails_root/public/image_trans/:id/:filename"
  validates :text_image, attachment_presence: true
  validates_attachment_content_type :text_image, :content_type => ["image/jpg", "image/jpeg", "image/png"]

  has_many :anns, dependent: :destroy
end
