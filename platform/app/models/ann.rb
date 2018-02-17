class Ann < ApplicationRecord
  belongs_to :image
  enum flag: [:ground, :eval]
end
