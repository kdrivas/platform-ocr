class CreateAnns < ActiveRecord::Migration[5.1]
  def change
    create_table :anns do |t|
      t.integer :flag, :default => 1
      t.integer :votes, :default => 1
      t.timestamps
    end
  end
end
