class CreateAnns < ActiveRecord::Migration[5.1]
  def change
    create_table :anns do |t|
      t.float :x0
      t.float :y0
      t.float :x1
      t.float :y1
      t.string :word
      t.integer :flag, :default => 1
      t.integer :votes, :default => 1
      t.timestamps
    end
  end
end
